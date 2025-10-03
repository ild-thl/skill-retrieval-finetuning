# Get all json files in the current directory and combine them into one file
import json
import os
import sys
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Combine JSON datasets from specified directories."
    )
    parser.add_argument(
        "directories", nargs="+", help="List of directories to search for JSON files"
    )
    parser.add_argument(
        "--verbose-duplicates",
        action="store_true",
        help="Show detailed logging for each duplicate merge",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="combined.json",
        help="Output filename for the combined dataset (default: combined.json)",
    )

    # Parse arguments
    args = parser.parse_args()
    # Directories are now set by command line parameters
    directories = args.directories
    verbose_duplicates = args.verbose_duplicates
    output_filename = args.output

    # List to hold all found JSON files
    json_files = []

    # Loop through each directory
    for directory in directories:
        # Construct the full path for the directory
        dir_path = os.path.join(".", directory)
        # Get all json files in the current directory
        files_in_dir = [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if file.endswith(".json")
        ]
        # Extend the json_files list with the files found in the current directory
        json_files.extend(files_in_dir)

    print(json_files)

    # Combine all json files into one file
    combined_json = []
    for js in json_files:
        with open(js, encoding="utf-8") as json_file:
            json_text = json.load(json_file)
            for item in json_text:
                # Clean item. Items should only have the properties query, pos, neg
                # Remove all other properties.
                new_item = {}
                if not "query" in item:
                    print("Item has no query property")
                    continue
                new_item["query"] = item["query"]
                if "pos" in item:
                    new_item["pos"] = item["pos"]
                else:
                    new_item["pos"] = []
                if "neg" in item:
                    new_item["neg"] = item["neg"]
                else:
                    new_item["neg"] = []
                combined_json.append(new_item)

    # Check for duplicates and merge pos/neg labels
    print(f"Initial combined dataset size: {len(combined_json)}")

    # Dictionary to track queries and their associated pos/neg labels
    query_dict = {}
    duplicate_count = 0
    merged_queries = []

    for item in combined_json:
        query_text = item["query"]

        if query_text in query_dict:
            # Duplicate found - merge pos and neg labels
            duplicate_count += 1
            existing_item = query_dict[query_text]

            # Merge pos labels (remove duplicates)
            merged_pos = list(set(existing_item["pos"] + item["pos"]))
            # Merge neg labels (remove duplicates)
            merged_neg = list(set(existing_item["neg"] + item["neg"]))

            # Update the existing item
            query_dict[query_text]["pos"] = merged_pos
            query_dict[query_text]["neg"] = merged_neg

            # Log the merge
            if verbose_duplicates:
                print(f"DUPLICATE MERGED: Query starting with '{query_text[:60]}...'")
                print(
                    f"  - Original pos labels: {len(existing_item['pos'])}, neg labels: {len(existing_item['neg'])}"
                )
                print(
                    f"  - Added pos labels: {len(item['pos'])}, neg labels: {len(item['neg'])}"
                )
                print(
                    f"  - Final pos labels: {len(merged_pos)}, neg labels: {len(merged_neg)}"
                )
            merged_queries.append(query_text[:60])

        else:
            # New query - add to dictionary
            query_dict[query_text] = item.copy()

    # Convert back to list format
    final_combined_json = list(query_dict.values())

    # Log summary of deduplication
    print(f"\n=== DEDUPLICATION SUMMARY ===")
    print(f"Original dataset size: {len(combined_json)}")
    print(f"Duplicates found and merged: {duplicate_count}")
    print(f"Final dataset size: {len(final_combined_json)}")
    print(f"Reduction: {len(combined_json) - len(final_combined_json)} samples")

    if merged_queries:
        print(f"\nMerged queries (showing first few):")
        for i, query_preview in enumerate(merged_queries[:5], 1):
            print(f"  {i}. {query_preview}...")
        if len(merged_queries) > 5:
            print(f"  ... and {len(merged_queries) - 5} more")

    # Write combined json file
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(final_combined_json, outfile, ensure_ascii=False, indent=4)

    print(f"\nCombined dataset saved to: {output_filename}")


if __name__ == "__main__":
    main()
