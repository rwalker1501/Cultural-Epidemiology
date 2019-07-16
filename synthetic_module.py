
import os
import target_module as tam
import population_data_module as pdm
from classes_module import Target
from main_module import MainProgram

def generate_synthetic_results(base_path):
    critical_distance = 1000
    clustering_on = True
    date_window = 10000

    target_list = tam.read_target_list_from_csv(base_path + '/tests/synth_targets2' )
    results_filename = 'synth_test.csv'
    
    synthetic_binary_path = base_path+'/population_data/synth2.npz'
    synthetic_info_path = base_path+'/population_data/synth2_info.txt'
    synthetic = pdm.load_population_data_source("Synthetic", synthetic_binary_path, synthetic_info_path)
    
    mm = MainProgram()
    mm.set_clustering(clustering_on)
    mm.set_date_window(date_window)
    mm.set_critical_distance(critical_distance)

    mm.generate_results(synthetic, target_list, base_path, 'synthetic')

def test_synthetic_data(base_path):
    generate_synthetic_results(base_path)
    base_synth_file = base_path + "results/synthetic/synth_base.csv"
    new_synth_file = base_path + "results/synthetic/synthetic_results.csv"
    base_synth = open(base_synth_file)
    new_synth = open(new_synth_file)

    base_synth_line = base_synth.readline()
    new_synth_line = new_synth.readline()
    base_synth_line = base_synth.readline()
    new_synth_line = new_synth.readline()

    line_num = 2
    changed = False

    print("\n\nChecking differences in synthetic files...")
    while base_synth_line != '' or new_synth_line != '':

        base_synth_line = base_synth_line.strip()
        new_synth_line = new_synth_line.strip()
        if base_synth_line != new_synth_line:
            if new_synth_line == '' and base_synth_line != '':
                print("Missing: ", " Line-%d" % line_num, base_synth_line)
            elif base_synth_line == '' and new_synth_line != '':
                print("Additional: ", "Line-%d" % line_num, new_synth_line)
            else:
                print("Changed:", "Line-%d" %  line_num, new_synth_line)
                print("\tOld: " + base_synth_line)
            print('\n')
            changed = True

        base_synth_line = base_synth.readline()
        new_synth_line = new_synth.readline()
        line_num += 1

    if not changed:
        print("No difference. Files are identical except for date.\n\n")
    else:
        print("FILES ARE NOT IDENTICAL. Check files and code for cause. \n\n")

    base_synth.close()
    new_synth.close()
    
if __name__ == '__main__':
    base_path = os.getcwd() + '/'
    test_synthetic_data(base_path)