********************************************************************************
* Set up                                                                       *
* Last modified Oct 17, 2024                                                   *
********************************************************************************

set type double 
set maxvar 30000
ssc install frameappend , replace
ssc install fs, replace



cd "${DIR}"
capture mkdir "temp"
global test_run = upper("$test_run") 
* Fix missing file extensions in raw data files
cd "${DIR}${raw_data_folder}"
fs *
global files = r(files)
foreach i in $files {
	local suf = substr("`i'", length("`i'") - 3, 4)
	if "`suf'" != ".zip" & "`suf'" !=  ".csv" {
		local nn = "`i'" + ".csv"
		!rename "`i'" "`nn'"
	}
}