#!/usr/bin/env python3
import csv

csv_file = '/home/sgram/Heatmap/AMNIST/Box/binary_Ding/outputs_amnist_cropped/training_log.csv'

times = []

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    
    for row in reader:
        if row:  # Skip empty rows
            try:
                times.append(float(row[-1]))  # Last column
            except ValueError:
                continue

avg_time = sum(times) / len(times)
total_time = sum(times)
min_time = min(times)
max_time = max(times)

print(f"Average time: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
print(f"Min time: {min_time:.2f} seconds")
print(f"Max time: {max_time:.2f} seconds")
print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
print(f"Number of rows: {len(times)}")
