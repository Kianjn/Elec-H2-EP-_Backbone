####
#### Coverage summary, printed as "(percentage) covered".
####
#### Useful for CI environments that just want a summary (eg a Gitlab setup).
####

using Coverage

folder_to_process = abspath(@__DIR__, "..", "..", "src")
processed = process_folder(folder_to_process)
covered_lines, total_lines = get_summary(processed)
percentage = covered_lines / total_lines * 100
print("""
\n\n\n
Code coverage: $(percentage)%
\n\n\n
""")

folders_to_clean = abspath(@__DIR__, "..", "..")
clean_folder(folders_to_clean) # Clean up .cov files