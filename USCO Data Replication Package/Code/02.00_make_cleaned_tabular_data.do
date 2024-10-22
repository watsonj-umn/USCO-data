********************************************************************************
* Transform and clean data                                                     *
* Last modified April 10, 2024                                                 *
********************************************************************************

clear all
cd "${DIR}/Processed Data/Long form data"
fs REG_*
cd "${DIR}"
global file_cnt = 1
global output = "\Processed Data\Wide form data"
foreach i in `r(files)' {
	dis "Starting file $file_cnt: `i'"
	display "$S_TIME  $S_DATE"
	use "${DIR}/Processed Data/Long form data/`i'", clear
	if _N > 1 {
		local file = "`i'"

		run "${DIR}${code_folder}/02.01_drop_unused_fields.do"
		display "Done with do_01 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"

		run "${DIR}${code_folder}/02.02_count_claim_auth.do"
		display "Done with do_02 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"

		run "${DIR}${code_folder}/02.03_reshape.do"
		display "Done with do_03 for file number $file_cnt: `file'"
		*save "Intermediate Data/temp1_wide_`i'", replace
		display "$S_TIME  $S_DATE"
		
		run "${DIR}${code_folder}/02.04_consul_names.do"
		display "Done with do_04 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"

		run "${DIR}${code_folder}/02.05_rename_vars.do"
		display "Done with do_05 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"

		run "${DIR}${code_folder}/02.06_organize_columns.do"
		display "Done with do_06 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"
		
		run "${DIR}${code_folder}/02.07_clean_text.do"
		display "Done with do_07 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"
		
		run "${DIR}${code_folder}/02.08_standardize_place_names.do"
		display "Done with do_08 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"
		
		run "${DIR}${code_folder}/02.09_cleanup_extra_vars.do"
		display "Done with do_09 for file number $file_cnt: `file'"
		display "$S_TIME  $S_DATE"
	}
	else {
		drop *
	}
	
	save "${DIR}${output}/`i'", emptyok replace
	global file_cnt = $file_cnt + 1
}