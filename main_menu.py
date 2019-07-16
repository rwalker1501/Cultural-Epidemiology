import os
import sys
import json
import target_module as tam
from copy import deepcopy
from os.path import isfile, join
from main_module import MainProgram
from classes_module import Target, PopulationData
from collections import OrderedDict


class Driver:
	def __init__(self):
		self.main_program = MainProgram()

	def run(self):
	
		print('')
		self.print_label('Epidemiology of culture data analysis v.1.0 Copyright (C) 2019  Richard Walker & Camille Ruiz')
		user_option = 0
		while user_option != '4':
			parameters_filename = self.main_program.get_parameters_filename();
			self.print_parameters(parameters_filename);
			
			print('-----------------------')
			print('1)    Read parameters file')
			print('2)    Change parameters')
			print('3)    Generate results')
			print('4)    Exit')
			user_option = raw_input("Choose an option: ")
			if user_option=='1':
				self.read_parameters_file();
			elif user_option=='2':
				self.change_parameters_file();
			elif user_option=='3':
				report = self.main_program.generate_results()
				print(report);
			elif user_option!='4':
				print("Invalid option. Try again.")


	def read_parameters_file(self):
		parameters_dir = self.main_program.get_parameters_folder();

		user_option = '0';
		selected_parameters_file = "";
		while user_option != '3':

			if selected_parameters_file == "":
				selected_parameters_file = self.main_program.get_parameters_filename();

			self.print_parameters(selected_parameters_file);

			print('-----------------------')
			print('1)    Select parameters file')
			print('2)    Save and exit')
			print('3)    Cancel')
			user_option = raw_input("Choose an option: ")
			if user_option=='1':
				selected_parameters_file = self.select_file_from_folder(parameters_dir);
			elif user_option=='2':
				self.main_program.set_parameters_filename(selected_parameters_file);
				return
			elif user_option!='3':
				print("Invalid option. Try again.");

	def change_parameters_file(self):
		parameters_filename = self.main_program.get_parameters_filename();
		parameters_folder = self.main_program.get_parameters_folder();
		parameters = self.get_parameters(parameters_filename);
		keys = self.main_program.key_order

		num_parameters = len(parameters);
		user_option = 0;
		while user_option != str(num_parameters + 2):

			self.print_label("Current Parameters");
			for i in range(0, num_parameters):
				key = keys[i];
				to_print = str(i+1) + ")  " + key + ": " + str(parameters[key])
				print(to_print);

			save_option = num_parameters + 1;
			cancel_option = num_parameters + 2;
			print(str(save_option) + ")  Save and Exit");
			print(str(cancel_option) + ")  Cancel");

			user_option = raw_input("Choose an option: ")
			try:
				option = int(user_option);
			except:
				print("Invalid option. Try again.")
				continue;
			
			if option == save_option:
				success = False;
				filename = self.main_program.get_parameters_filename();
				while not success:
					filename = raw_input("Input parameters filename: ")
					filenames = [f for f in os.listdir(parameters_folder) if isfile(join(parameters_folder,f))];
					if filename in filenames:
						overwrite_option = "";
						while overwrite_option != 'y' and overwrite_option != "n":
							overwrite_option = raw_input("Overwrite file " + filename + "? [y/n]: ")
							if overwrite_option == 'y':
								success = True;
					else:
						success = True;

				print("Saving parameters...")
				filepath = os.path.join(parameters_folder, filename);
				self.save_parameters(filepath, parameters);
				self.main_program.set_parameters_filename(filename);
				user_option = str(cancel_option);

			elif option < save_option and option > 0:
				key = keys[option-1];
				if key == "globals_type":
					globals_types = ["All", "No equatorials", "France and Spain", "Australia"];
					success = False;

					while not success:
						print('----------------------');
						for i in range(0, len(globals_types)):
							print(str(i+1) + ")  " + globals_types[i]);
						globals_option = raw_input("Choose a globals type: ");
						try:
							globals_option = int(globals_option);
							parameters[key] = globals_types[globals_option-1];
							success = True;
						except:
							success = False;
				elif key == "target_file":
					targets_folder = self.main_program.get_targets_folder();
					parameters[key] = self.select_file_from_folder(targets_folder)[:-4];
				else:
					value = parameters[key];
					value_type = type(value);
					print("-------------------")
					print(key + ": " + str(value))
					print(str(value_type));
					print("-------------------")

					success = False;

					while not success:
						try:
							new_value = raw_input("Input new value: ")
							if isinstance(value, bool):
								if new_value.lower() == "true":
									new_value = True;
								elif new_value.lower() == "false":
										new_value = False;
								else:
									print("Wrong type. Try again.")
									continue
							else:
								new_value = value_type(new_value);
							parameters[key] = new_value;
							print("Value changed to: " + str(new_value));
							success = True;
						except:
							print("Wrong type. Try again.");
							success = False
			elif option != cancel_option:
				print("Invalid option. Try again.")

	def get_parameters(self, parameters_filename):
		parameters_folder = self.main_program.get_parameters_folder();
		parameters_filepath = os.path.join(parameters_folder, parameters_filename)
		return json.load(open(parameters_filepath))

	def print_parameters(self, parameters_filename):
		parameters = self.get_parameters(parameters_filename);
		keys = self.main_program.key_order
		
		self.print_label("Current Parameters");
		for key in keys:
			print(key + ": " + str(parameters[key]));	

		print("\nParameters file: " + parameters_filename);

	def save_parameters(self, filepath, data):
		ordered = [];
		keys = self.main_program.key_order
		for key in keys:
			ordered.append((key,data[key]));
		ordered = OrderedDict(ordered);
		with open(filepath, "w") as write_file:
			json.dump(ordered, write_file, indent=4);


	def select_file_from_folder(self, folder):
		filenames = [f for f in os.listdir(folder) if isfile(join(folder,f))];
		print("***********************")
		print("*** AVAILABLE FILES ***")
		print("***********************")
		for name in filenames:
			print(name)
		print("***********************")
		filename = "";
		while filename not in filenames:
			filename = raw_input("Select and input filename: ");
		return filename;

	def print_label(self, label):
		asterisks = ''
		for i in range(0, len(label)+2):
			asterisks += "*"
		print(asterisks)
		print(label)
		print(asterisks)

if __name__ == '__main__':
	dr = Driver()
	dr.run();