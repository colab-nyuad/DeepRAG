import json
import jsonlines

def process_line(item):
    # Remove expansion_info field if it exists
    if 'expansion_info' in item:
        del item['expansion_info']
    return item

def main():
    input_file = 'sub2.jsonl'
    output_file = 'sub2_no_expansion.jsonl'
    
    with jsonlines.open(input_file) as reader, \
         jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            processed = process_line(obj)
            writer.write(processed)
    
    print(f"Processed {input_file} -> {output_file}")

if __name__ == "__main__":
    main() 