import time
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from FlagEmbedding import FlagReranker
from api_client import APIClient
from api_config import APIConfig


class RetrievalEvaluator:
    def __init__(
        self,
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        store_docs: List[Document],
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
    ) -> None:
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)
        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.store_docs = store_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.total_query_chars = sum(len(value) for value in self.queries)

        self.db_client = chromadb.PersistentClient("./tmp/chroma_db_temp_no_sync")

    def load_reranker(self, model_name: str) -> FlagReranker:
        return FlagReranker(
            model_name, use_fp16=True
        )  # use fp16 can speed up computing

    def create_vector_store(
        self,
        model_name: str,
        embed_instruction: Optional[str] = "",
        use_cached_db: bool = False,
    ) -> Chroma:
        # Define the collection name
        collection_name = "temp"
        # Initialize the embedding function
        if "instruct" in model_name:
            embedding_function = HuggingFaceInstructEmbeddings(
                model_name=model_name,
                query_instruction="Represent the learning outcome for retrieving relevant skills: ",
                embed_instruction="Represent the skill for retrieval: ",
            )
        else:
            embedding_function = HuggingFaceEmbeddings(model_name=model_name)

        # Check if the collection exists
        collection = self.db_client.get_or_create_collection(name=collection_name)
        is_collection_set = collection.count() > 0

        # If caching is not desired, delete the existing collection
        if not use_cached_db:
            print("Deleting existing vector store.")
            self.db_client.delete_collection(name=collection_name)
            is_collection_set = False  # Ensure the collection is marked as not set

        # Create or return the Chroma vector store based on the collection's existence
        if is_collection_set:
            # Return an existing Chroma instance if the collection is already set
            db = Chroma(
                client=self.db_client,
                collection_name=collection_name,
                embedding_function=embedding_function,
                client_settings=Settings(anonymized_telemetry=False),
                collection_metadata={"hnsw:space": "cosine"},
            )

            print(
                "Using existing vector store containing",
                db._collection.count(),
                "documents.",
            )
            return db
        else:
            # Copy self.store_docs to avoid modifying the original list
            self.store_docs = self.store_docs.copy()
            # Add embedding instructions to the documents page_content
            for doc in self.store_docs:
                doc.page_content = embed_instruction + doc.page_content

            # Create a new Chroma instance with documents if the collection is not set or was deleted
            def split_docs(docs, batch_size):
                for i in range(0, len(docs), batch_size):
                    yield docs[i : i + batch_size]

            doc_batches = split_docs(self.store_docs, 4000)
            batch_index = 0  # Initialize a counter to track the batch number
            batch_count = (
                len(self.store_docs) // 4000
            )  # Calculate the total number of batches
            # + 1 if there is a rest
            if len(self.store_docs) % 4000 > 0:
                batch_count += 1

            for batch in doc_batches:
                batch_index += 1  # Increment the counter after processing each batch
                print(f"Embedding batch: {batch_index} of {batch_count}", end="\r")
                db = Chroma.from_documents(
                    client=self.db_client,
                    collection_name=collection_name,
                    documents=batch,  # Use the current batch instead of self.store_docs
                    embedding=embedding_function,
                    collection_metadata={"hnsw:space": "cosine"},
                    client_settings=Settings(anonymized_telemetry=False),
                )

            print(
                "Created vector store containing", db._collection.count(), "documents."
            )
            return db

    def compute_metrics(
        self, queries_result_list: Dict[str, List[object]], total_time: float
    ) -> Dict[str, Dict[int, float]]:
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_id in self.queries_ids:
            # Sort scores
            top_hits = sorted(
                queries_result_list[query_id], key=lambda x: x["score"], reverse=True
            )
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0
                    for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(
                    predicted_relevance, k_val
                ) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        metrics = {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
            "total_time": total_time,
            "avg_time_per_query": total_time / len(self.queries),
            "avg_time_per_1000_chars": (total_time / self.total_query_chars) * 1000,
        }

        metrics_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_dict[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    new_key = k.replace("@k", f"@{kk}")
                    metrics_dict[new_key] = vv

        return metrics_dict

    def rerank(
        self, query: str, predictions: List[Tuple[str, float]], reranker
    ) -> List[Tuple[str, float]]:
        pairs = [(query, p[0]) for p in predictions]
        scores = reranker.compute_score(pairs)
        # Convert scores to list if necessary.
        if not isinstance(scores, list):
            scores = [scores]

        # Reranked predictions with positive scores.
        reranked = []
        for p, score in zip(predictions, scores):
            # Normalize score to be between 0 and 1.
            # max_score = 13
            # score = max(min(score, max_score), -max_score)
            # score = (score + max_score) / (max_score * 2)
            reranked.append((p[0], score))

        reranked = sorted(reranked, key=lambda p: p[1], reverse=True)
        return reranked

    def predict(
        self,
        query: str,
        db: Chroma,
        reranker: Optional[FlagReranker] = None,
        query_instruction: Optional[str] = "",
    ) -> List[Tuple[str, float]]:
        top_k = min(30, len(self.corpus))
        predictions = db.similarity_search_with_relevance_scores(
            query_instruction + query, top_k
        )
        predictions = [(p[0].metadata["title"], p[1]) for p in predictions]
        if reranker:
            predictions = self.rerank(query, predictions, reranker)
        return predictions

    def predict_api(
        self, query: str, api_client: APIClient, top_k: int = 30
    ) -> List[Tuple[str, float]]:
        """Make predictions using an API client"""
        try:
            predictions = api_client.predict(query, top_k=top_k)
            # Filter predictions to only include skills that exist in our corpus
            filtered_predictions = []
            for skill_name, score in predictions:
                if skill_name in self.corpus:
                    filtered_predictions.append((skill_name, score))
            return filtered_predictions
        except Exception as e:
            print(f"API prediction failed for query '{query[:50]}...': {str(e)}")
            return []

    def __call__(
        self,
        embedding_model_name: str = None,
        reranker_model_name: Optional[str] = None,
        query_instruction: Optional[str] = "",
        embed_instruction: Optional[str] = "",
        api_config: Optional[APIConfig] = None,
        use_cached_db: bool = False,
    ):
        if api_config:
            # API-based evaluation
            return self.evaluate_api(api_config)
        else:
            # Model-based evaluation (existing functionality)
            return self.evaluate_models(
                embedding_model_name,
                reranker_model_name,
                query_instruction,
                embed_instruction,
                use_cached_db,
            )

    def evaluate_models(
        self,
        embedding_model_name: str,
        reranker_model_name: Optional[str] = None,
        query_instruction: Optional[str] = "",
        embed_instruction: Optional[str] = "",
        use_cached_db: bool = False,
    ):
        """Evaluate using embedding models and rerankers (original functionality)"""
        db = self.create_vector_store(
            embedding_model_name,
            embed_instruction=embed_instruction,
            use_cached_db=use_cached_db,
        )
        reranker = None
        if reranker_model_name:
            reranker = self.load_reranker(reranker_model_name)

        total_time = 0.0

        queries_result_list = {}
        # Iterate through stored queries IDs and texts
        for query_id, query in zip(self.queries_ids, self.queries):
            # Compute predictions
            start = time.time()
            predictions = self.predict(
                query, db, reranker, query_instruction=query_instruction
            )
            total_time += time.time() - start

            query_results = []
            for p in predictions:
                corpus_id = self.corpus.index(p[0])
                query_results.append({"corpus_id": corpus_id, "score": p[1]})

            queries_result_list[query_id] = query_results

            # show progress
            print(f"Progress: {len(queries_result_list)}/{len(self.queries)}", end="\r")

        return self.compute_metrics(queries_result_list, total_time)

    def evaluate_api(self, api_config: APIConfig):
        """Evaluate using an API"""
        api_client = APIClient(api_config)
        total_time = 0.0

        queries_result_list = {}
        failed_queries = 0

        # Rate limiting
        import time as time_module

        time_between_requests = 1.0 / api_config.max_requests_per_second

        # Iterate through stored queries IDs and texts
        for query_id, query in zip(self.queries_ids, self.queries):
            # Rate limiting
            start = time_module.time()

            try:
                predictions = self.predict_api(query, api_client, top_k=30)

                query_results = []
                for skill_name, score in predictions:
                    if skill_name in self.corpus:
                        corpus_id = self.corpus.index(skill_name)
                        query_results.append({"corpus_id": corpus_id, "score": score})

                queries_result_list[query_id] = query_results

            except Exception as e:
                print(f"Failed to get predictions for query {query_id}: {str(e)}")
                failed_queries += 1
                queries_result_list[query_id] = []  # Empty results for failed queries

            elapsed = time_module.time() - start
            total_time += elapsed

            # Rate limiting sleep
            sleep_time = max(0, time_between_requests - elapsed)
            if sleep_time > 0:
                time_module.sleep(sleep_time)

            # Show progress
            print(
                f"Progress: {len(queries_result_list)}/{len(self.queries)} (Failed: {failed_queries})",
                end="\r",
            )

        print(
            f"\nAPI evaluation completed. Failed queries: {failed_queries}/{len(self.queries)}"
        )

        return self.compute_metrics(queries_result_list, total_time)

    @staticmethod
    def compute_dcg_at_k(relevances: List[int], k: int) -> float:
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
