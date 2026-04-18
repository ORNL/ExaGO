#!/usr/bin/python

import sys


def header(bus_size):
    output = []
    output.append("function mpc = CICJ_unittestx" + bus_size + "\n")
    output.append("mpc.version = '2';\n")
    output.append("mpc.baseMVA =  100.00;\n")
    output.append("\n")
    return output


def bus(bus_size):
    output = []
    output.append("%% bus data\n")
    output.append("mpc.bus = [\n")
    output.append( "1 1  0.00 0.00 0.00  0.00 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
    output.append( "2 1  0.00 0.00 0.25 -0.05 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
    output.append( "3 3  0.00 0.00 0.00  0.00 1 0.5000000 30.000000  13.80 1 1.100 0.900\n")
    output.append( "4 1 -3.40 8.00 0.00  0.00 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
    output.append( "5 1  0.00 0.00 0.00  0.00 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
    for i in range(1, int(bus_size)):
        bus_idx = (i + 1) * 4
        output.append(f"{bus_idx - 2} " +
            "1  0.00 0.00 0.25 -0.05 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
        output.append(f"{bus_idx - 1} " +
            "2  0.00 0.00 0.00  0.00 1 0.5000000 30.000000  13.80 1 1.100 0.900\n")
        output.append(f"{bus_idx} " +
            "1 -3.40 8.00 0.00  0.00 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
        output.append(f"{bus_idx + 1} " +
            "1  0.00 0.00 0.00  0.00 1 2.0000000  0.000000 138.00 1 1.100 0.900\n")
    output.append("];\n")
    output.append("\n")
    return output


def generator(bus_size):
    output = []
    output.append("%% generator data\n")
    output.append("mpc.gen = [\n")
    for i in range(3, int(bus_size) * 4, 4):
        output.append(
            f"{i} 1.60 -2.20 197.80 -202.20 0.5000 100.00 1 1.60 1.60 0.00 0.00 0.00 0.00 0.00 0.00 0 0 0 10.0000\n")
    output.append("];\n")
    output.append("\n")
    return output


def gen_cost(bus_size):
    output = []
    output.append("%% generator cost data\n")
    output.append("mpc.gencost = [\n")
    for _ in range(int(bus_size)):
        output.append("2 0 0 3 0.010 0.100 8.00 0.0000 \n")
    output.append("];\n")
    output.append("\n")
    return output


def branch(bus_size):
    output = []
    output.append("%% branch data\n")
    output.append("mpc.branch = [\n")
    for i in range(int(bus_size)):
        bus_idx = (4 * i + 1)
        output.append(f"{bus_idx} {bus_idx + 1} " +
            "2.000000 1.000000 1.20000 230.00 230.00 230.00 1.00000  0.000 1 0.00 0.00 0.00 -240.00   0.00 -240.00\n")
        output.append(f"{bus_idx + 1} {bus_idx + 2} " +
            "2.000000 1.000000 1.20000 230.00 230.00 230.00 2.00000 60.000 1 0.00 0.00 0.00 -140.00 200.00 -240.00\n")
        output.append(f"{bus_idx + 1} {bus_idx + 3} " +
            "2.000000 1.000000 1.20000 230.00 230.00 230.00 1.00000  0.000 1 0.00 0.00 0.00 -240.00   0.00 -240.00\n")
        output.append(f"{bus_idx + 3} {bus_idx + 4} " +
            "2.000000 1.000000 1.20000 230.00 230.00 230.00 1.00000  0.000 1 0.00 0.00 0.00 -240.00   0.00 -240.00\n")
    output.append("];\n")
    output.append("\n")
    return output


def bus_names(bus_size):
    output = []
    output.append("%% bus names\n")
    output.append("mpc.bus_name = {\n")
    for i in range(int(bus_size) * 4 + 1):
        output.append(f"'{i + 1}';\n")
    output.append("};\n")
    output.append("\n")
    return output


def gen_types(bus_size):
    output = []
    output.append("%% Generator Unit Types\n")
    output.append("mpc.gentype = {\n")
    for i in range(int(bus_size)):
        output.append("'UN';\n")
    output.append("};\n")
    output.append("\n")
    return output


def gen_fuel(bus_size):
    output = []
    output.append("%% Generator Fuel Types\n")
    output.append("mpc.genfuel = {\n")
    for i in range(int(bus_size)):
        output.append("'unknown';\n")
    output.append("};\n")
    return output


if __name__ == '__main__':
    network_length = 0

    # Script can be called with CLI argument
    if (len(sys.argv) < 2):
        network_length = input('Enter a network size: ')
    else:
        network_length = sys.argv[1]

    print(f'Generating a network of size {network_length}')

    output_filename = 'CICJ_unittestx' + network_length + '.m'

    f = open(output_filename, "w+")

    # Loop over file component functions and append to file
    for section in [header, bus, generator, gen_cost, branch, bus_names, gen_types, gen_fuel]:
        for line in section(network_length):
            f.write(line)

    f.close()
