import sys
from pathlib import Path

import shutil
import os


def copy_file(file_path_str, dest_dir):
    p = Path(file_path_str)
    if p.is_file():
        try:
            shutil.copy(file_path_str, dest_dir)
        except FileNotFoundError:
            print(f"Error: The source file '{file_path_str}' was not found or the destination directory is invalid.")
        except PermissionError:
            print(f"Error: Permission denied. Check write permissions for the destination directory.")
        except shutil.SameFileError:
            pass
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return dest_dir + f"/{p.stem}{p.suffix}"
    else:
        return None


filename = sys.argv[1]
destdir = sys.argv[2]
os.makedirs(destdir, exist_ok=True)
vizsrcdir = sys.argv[3]

filepath = copy_file(filename, destdir)
if filepath is None:
    print("Error: File does not exist.")
    sys.exit(1)

with open(vizsrcdir + "/module_casedata.js", "w") as f:
    f.write("// ExaGo Viz Input File\n")
    f.write("\n")
    f.write('import inputcasedata from "' + filepath + '" with { type: "json" };\n')
    f.write("\n")
    f.write("export default {\n")
    f.write("  get_casedata() {\n")
    f.write("    var casedata0 = {};\n")
    f.write("\n")
    f.write("    casedata0.geojsondata = {};\n")
    f.write("    casedata0.nareas = inputcasedata.nareas;\n")
    f.write("    casedata0.nzones = inputcasedata.nzones;\n")
    f.write("    casedata0.areas = inputcasedata.areas;\n")
    f.write("    casedata0.zones = inputcasedata.zones;\n")
    f.write('    casedata0.geojsondata.type = "FeatureCollection";\n')
    f.write("    casedata0.geojsondata.features = [...inputcasedata.geojsondata.features];\n")
    f.write("    return casedata0;\n")
    f.write("  },\n")
    f.write("};\n")
