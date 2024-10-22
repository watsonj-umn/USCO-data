********************************************************************************
* Make long-form categorized data files                                        *
* Last modified Oct 17, 2024                                                   *
********************************************************************************

clear
clear frames
global output_folder = "\temp\parsed uncat data"
global max_filesize = 16 

frame rename default long_data
capture mkdir "${DIR}${output_folder}"
cd "${DIR}${raw_data_folder}"
fs *.csv
cd "${DIR}
global itt = 0
foreach f in `r(files)' {
	global itt = $itt + 1
	display "Starting file $itt"
	global file_name = "`f'" 
	global name = "\" + substr("$file_name", 1 , length("$file_name")-4)
	
	* Import text as single column by preventing Stata from mistaking certain characters as delimiters

	if "$test_run" == "TRUE" {
		dis "Importing only first 100,000 rows for testing"
		qui: import delimite `"${DIR}${raw_data_folder}\\${file_name}"',   delimiter("TreatAsSingleColumn", asstring) rowrange(  : 100000) clear
	}
	else {
		qui: import delimite `"${DIR}${raw_data_folder}\\${file_name}"',   delimiter("TreatAsSingleColumn", asstring) clear
	}

	* Create a record ID for resahaping data from long form to wide form
	run `"${DIR}${code_folder}\01.01_Row_ID.do"'

	* Fix the small number of entries that eroniously span multiple rows.
	run `"${DIR}${code_folder}\01.02_Fix_Row_Split_Errors.do"'

	* Parse text into data and variable names in batches
	global max_records = _N
	global rec_per_it = min( _N , 500000 )
	global max_itts = ceil( $max_records / $rec_per_it )
	capture frame drop temp_large
	frame create temp_large
	global filecount = 1
	global record_count = 0
	global i = 1
	while $max_records > $record_count {
		local i = $i
		run `"${DIR}${code_folder}\01.03_Name_Variables.do"'
		display as result "Done with `i' of $max_itts variable name subsets"
	}

	* Rejoin batches into single file or set of files, depending on the max output 
	* file size specified above.
	use `"${DIR}\temp\temp_1"', clear
	global filecount = ceil($max_itts / 10)
	qui: des
	global filesize = r(width) * r(N)/1000000000
	global part_count = 0
	global new_itt = 0
	if $filecount > 1 {
		forvalues i = 2/$filecount {
			global new_itt = 0
			if $filesize > $max_filesize * 0.9 {
				global part_count = $part_count + 1
				sort obs_num
				save "${DIR}${output_folder}${name}_ln_data_part_$part_count"
				global new_itt = 1
				clear
			}
			append using `"${DIR}\temp\temp_`i'"'
			erase `"${DIR}\temp\temp_`i'.dta"'
			qui: des
			global filesize = r(width) * r(N)/1000000000
		}
	}
	sort obs_num
	format v_ %35s
	if $part_count > 0  & $new_itt == 0 {
		global part_count = $part_count + 1
		save "${DIR}${output_folder}${name}_ln_data_part_$part_count" , replace
		erase  `"${DIR}\temp\temp_1.dta"'
	}
	else if $part_count == 0  {
		save "${DIR}${output_folder}${name}_ln_data" , replace
		erase  `"${DIR}\temp\temp_1.dta"'
	}
}

run `"${DIR}${code_folder}\01.04_Create_categorized_files.do"'