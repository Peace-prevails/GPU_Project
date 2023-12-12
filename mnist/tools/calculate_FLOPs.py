import sys

def parse_metric_value(line):
    """ Extracts and returns the numeric value from a line in the log file. """
    return int(line.split()[-1].replace(',', ''))

def calculate_sum_from_log(file_path):
    fadd_sum = fmul_sum = ffma_sum = 0

    with open(file_path, 'r') as file:
        for line in file:
            if 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum' in line:
                fadd_sum += parse_metric_value(line)
            elif 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum' in line:
                fmul_sum += parse_metric_value(line)
            elif 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum' in line:
                ffma_sum += parse_metric_value(line)

    # Perform the calculation
    total = fadd_sum + fmul_sum + ffma_sum * 2
    return total


if __name__ == "__main__":
    # Get file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python calculate_FLOPs.py <log_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    total_sum = calculate_sum_from_log(file_path)
    print(f'Total Sum: {total_sum}')

