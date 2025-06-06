# main.py

from utils import run_tests_and_generate_reports, run_all_providers_and_generate_reports
from report import generate_root_summary


#def run_tests_and_generate_reports(provider="DnnlExecutionProvider",
#                                   test_file="test.txt",
#                                   profiling_dir="profiling",
#                                   models_dir="models"):


# Simply call the function with default provider "CUDAExecutionProvider"
#run_tests_and_generate_reports("DmlExecutionProvider")
#print("Done: tests completed and both reports generated.")

#def run_all_providers_and_generate_reports(test_file="test.txt",
#                                           base_profiling_dir="profiling",
#                                           base_models_dir="models"):

# Simply call the function to execute with all available Execution Providers thanks to get_available_providers()
run_all_providers_and_generate_reports("test.txt", "profiling", "models")
print("Done: tests completed and both reports generated.")


# Generate a ReadMe summarising the results of each EP already carried out.
generate_root_summary()