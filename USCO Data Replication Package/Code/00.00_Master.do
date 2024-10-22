********************************************************************************
* Transform and clean copyright data                                           *
* Last modified Oct 17, 2024                                                   *
********************************************************************************
clear all
clear frames

********************************************************************************
* NOTES:
* Raw data files must be downloaded and unzipped into "Raw Data" folder - 
* For raw data go to https://copyright.gov/economic-research/usco-datasets/ 
* and select "Raw Unparsed Uncategorized XML" or initiate download using 
* https://data.copyright.gov/Raw%20unparsed%20uncategorized%20XML/USCO%20public%20data.zip

* Enter root directory (ending with "\Data replication package")
  global DIR = [Enter working directory here]

* To test code with small subset of date enter "TRUE"; otherwise to run normally enter "FALSE"
  global test_run = "TRUE"
********************************************************************************
capture log close
log using `"${DIR}\Log\log ${S_DATE}"' , replace

cd "${DIR}"
global raw_data_folder = "/Raw Data"
global code_folder = "/Code"

run "${DIR}${code_folder}/00.01_Set_up.do"
run "${DIR}${code_folder}/01.00_make_long_form_data_from_xml_files.do"
run "${DIR}${code_folder}/02.00_make_cleaned_tabular_data.do"
capture log close
cd "${DIR}"
shell rmdir "${DIR}\temp" /s /q