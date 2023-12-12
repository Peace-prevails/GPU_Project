import sys

# Convert values to bytes based on unit
def convert_to_bytes(value, unit):
    if unit == 'byte':
        return value  # Already in bytes
    elif unit == 'Kbyte':
        return value * 1024  # Convert kilobytes to bytes
    elif unit == 'Mbyte':
        return value * 1024 * 1024  # Convert megabytes to bytes
    else:
        raise ValueError("Unknown unit for conversion")

# convert values to megabytes based on unit
def convert_to_megabytes(value, unit):
    if unit == 'byte':
        return value / (1024 * 1024)  # Convert bytes to megabytes
    elif unit == 'Kbyte':
        return value / 1024  # Convert kilobytes to megabytes
    elif unit == 'Mbyte':
        return value
    else:
        raise ValueError("Unknown unit for conversion")

def calculate_sum_from_log(file_path):
    # log file path

    # Initialize both read and write bytes
    total_dram_read_bytes_sum = 0.0
    total_dram_write_bytes_sum = 0.0

    with open(file_path, 'r') as file:
        for line in file:
            if 'fbpa__dram_read_bytes.sum' in line:
                parts = line.split()
                value, unit = float(parts[-1]), parts[-2]
                total_dram_read_bytes_sum += convert_to_bytes(value, unit)
            elif 'fbpa__dram_write_bytes.sum' in line:
                parts = line.split()
                value, unit = float(parts[-1]), parts[-2]
                total_dram_write_bytes_sum += convert_to_bytes(value, unit)
    return total_dram_read_bytes_sum, total_dram_write_bytes_sum


if __name__ == "__main__":
    # Get file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python calculate_dram.py <log_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    total_dram_read_bytes_sum, total_dram_write_bytes_sum = calculate_sum_from_log(file_path)
    print(f'Total DRAM Read Bytes: {total_dram_read_bytes_sum} bytes')
    print(f'Total DRAM Write Bytes: {total_dram_write_bytes_sum} bytes')