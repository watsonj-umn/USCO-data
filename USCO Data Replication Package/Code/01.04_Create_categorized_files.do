********************************************************************************
* Divide data by type of work/record                                           *
* Last modified Oct 17, 2024                                                   *
********************************************************************************
clear
clear frames
global output_folder = "\Processed Data\Long form data"
global input_folder = "\temp\parsed uncat data"
cd `"${DIR}${input_folder}"'
fs *
global files = r(files)
global num_fi = wordcount(`"$files"')
global start_fn = 1
global end_fn = 26
cd `"${DIR}\temp"'
capture rmdir "Index"
capture mkdir "Index"
capture mkdir "${DIR}\temp\Index\index per file"
capture mkdir "${DIR}\temp\Long Data"
capture mkdir "${DIR}\temp\Long Data\with cat"
cd `"${DIR}\temp\Index"'
frame create hold
global itt = 0
foreach j in $files {
	use `"${DIR}${input_folder}/`j'"', clear
	sort obs_num
	keep if  var_nm == "leader_1" | field_number == "017" |  field_number == "017" | field_number == "027" | field_number == "917"
	gen a = length(var_nm)
	gen b = substr(var_nm, a - 1, 2)
	keep if  b == "_1"
	drop b
	gen b = substr(var_nm, a - 2, 1)
	replace field_number = field_number + "_" + b
	drop b
	keep if var_nm == "leader_1" | field_number == "017" |  field_number == "017_a" | field_number == "027_a" | field_number == "917_b" | field_number == "917_c"
	replace field_number = "leader" if var_nm == "leader_1"
	drop obs_num var_nm a
	gen a = substr(v_, 7,2) if field_number == "leader" 
	replace a = substr(v_, 1, 3) if field_number == "017_a" & substr(v_, 3, 1) != "0"
	replace a = substr(v_, 1, 2) if field_number == "017_a" & substr(v_, 3, 1) == "0"
	replace a = substr(v_, 1, 3) if field_number == "027_a" & substr(v_, 3, 1) != "0"
	replace a = substr(v_, 1, 2) if field_number == "027_a" & substr(v_, 3, 1) == "0"	
	replace v_ = a if a != ""
	drop a
	gen file = "`j'"
	replace file = substr(file,30,2)
	replace file = substr(file,1,1) if substr(file,2,1) == "_"
	destring file, replace	
	encode field_nu,  gen(temp1)
	drop field_nu
	rename temp1 field_number
	gen a = substr(v_, 2, 1)
	gen b = substr(v_, 1, 1) 
	gen c = "V" if b == "V" & ( a == "1" | a == "2" | a == "3" |a == "4" |a == "5" |a == "6" |a == "7" |a == "8" |a == "9" | a == "0")
	replace v_ = c if c != ""
	drop a b c
	drop if v_ == "ab"
	drop if v_ == "12"
	drop if v_ == "123"
	drop if v_ == "124"
	drop if v_ == "126"
	drop if v_ == "129"
	drop if v_ == "13"
	drop if v_ == "141"
	drop if v_ == "149"
	drop if v_ == "151"
	drop if v_ == "154"
	drop if v_ == "157"
	drop if v_ == "163"
	drop if v_ == "164"
	drop if v_ == "17"
	drop if v_ == "171"
	drop if v_ == "194"
	drop if v_ == "199"
	drop if v_ == "20"
	drop if v_ == "201"
	drop if v_ == "213"
	drop if v_ == "166"
	replace v_ = "V" if v_ == "V2"
	replace v_ = "VA" if v_ == "VA " | v_ == "Va" | v_ == "VAe 5" | v_ == "VH"
	replace v_ = "W Cancellation" if v_ == "W  Cancellation" | v_ == "W cancellation" | v_ == "B W Cancellation"
	replace v_ = "X Motion picture or filmstrip" if v_ == "X" | v_ == "X Motion Picture"  | v_ == "X Motion Picture or filmstrip"  | v_ == "X Motion picture"
	replace v_ = "S Art work" if v_ == "S Art Work"
	replace v_ = "SR" if v_ == "SR "
	replace v_ = "TX" if v_ == "TX " | v_ == "TX0006979991"
	replace v_ = "U Sound recording" if v_ == "U Sound Recording"
	replace v_ = "rc" if v_ == "re-"
	replace v_ = "1 Architectural drawing" if v_ == "1 Architectural dra0ing"
	replace v_ = "6 Serial" if substr(v_,1,1) == "6" | v_ == "B 6 serial"
	replace v_ = "RE" if v_ == "RE " | v_ == "Re-"
	replace v_ = "N Sound recording and music" if v_ == "N Sound Recording and music" | v_ == "N Sound recording and music"
	replace v_ = "M Musical work" if v_ == "M Musical Work" | v_ == "M Musical work" | v_ == "MW"
	replace v_ = "D Dramatic work and accompanying music" if v_ == "D Dramatic work and accompanying music, if any" | v_ == "D Dramatic Work and accompanying music"
	replace v_ = "C Machine-readable work or computer program" if v_ == "C Machine readable work or computer program"
	tostring file, g(a)
	tostring row_id, g(b)
	gen fr = a + "_" + b
	drop a b 
	drop if row_id == 650486 & file == 16
	replace v_ = "SR" if (row_id == 136636 & file == 17 & field_number == 1) | (row_id == 306424 & file == 18 & field_number == 1)
	replace v_ = "PA" if (row_id == 306427 & file == 18 & field_number == 1) 
	replace v_ = "TX" if (row_id == 381034 & file == 18 & field_number == 1) 
	drop if row_id == 79120 & file == 15
	replace v_ = "RE" if v_ == "EF" | v_ == "EU"
	replace v_ = "TXu" if v_ == "rc"
	drop if v_ == "nR5"  & field_number == 1
	drop if v_ == "V"  & field_number == 1
	replace v_ = "TX" if (row_id == 991674 & file == 9 & field_number == 1) 
	replace v_ = "TX" if (row_id == 147375 & file == 10 & field_number == 1) 
	replace v_ = "TX" if (row_id == 163307 & file == 10 & field_number == 1) 
	replace v_ = "TX" if (row_id == 257660 & file == 10 & field_number == 1) 
	replace v_ = "TX" if (row_id == 1004879 & file == 10 & field_number == 1) 
	replace v_ = "PA" if (row_id == 975017 & file == 6 & field_number == 1) 
	replace v_ = "PA" if (row_id == 643307 & file == 7 & field_number == 1)
	replace v_ = "VA" if v_ == "Rer"
	drop if v_ == "Cho"

	cd "${DIR}\temp"
	frame put if field_number == 1 , into(temp_017) 
	frame change temp_017
	sort fr
	drop if fr == fr[_n+1]
	save "${DIR}\temp\temp_017", replace
	frame change default 
	frame drop temp_017

	frame put if field_number == 2 , into(temp_027) 
	frame change temp_027
	drop if fr == fr[_n+1]
	save "${DIR}\temp\temp_027", replace
	frame change default 
	frame drop temp_027

	frame put if field_number == 3 , into(temp_917_b) 
	frame change temp_917_b
	drop if fr == fr[_n+1]
	save "${DIR}\temp\temp_917_b", replace
	frame change default 
	frame drop temp_917_b

	frame put if field_number == 4 , into(temp_917_c) 
	frame change temp_917_c
	drop if fr == fr[_n+1]
	save "${DIR}\temp\temp_917_c", replace
	frame change default 
	frame drop temp_917_c

	frame put if field_number == 5 , into(temp_leader) 
	frame change temp_leader
	drop if fr == fr[_n+1]
	save "${DIR}\temp\temp_leader", replace
	frame change default 
	frame drop temp_leader

	use "${DIR}\temp\temp_leader.dta", clear
	rename v_ f_leader
	merge 1:1 fr using "${DIR}\temp\temp_017.dta"
	drop _merge
	rename v_ f_017
	merge 1:1 fr using "${DIR}\temp\temp_027.dta"
	drop _merge
	rename v_ f_027
	merge 1:1 fr using "${DIR}\temp\temp_917_b.dta"
	drop _merge
	rename v_ f_917_b
	merge 1:1 fr using "${DIR}\temp\temp_917_c.dta"
	drop _merge
	rename v_ f_917_c

	drop field_n
	order file row_id f_leader f_017 f_027 f_917_b f_917_c
	sort file row_id

	replace f_leader = "Missing" if f_leader == ""
	replace f_017 = "Missing" if f_017 == ""
	replace f_027 = "Missing" if f_027 == ""
	replace f_917_b = "Missing" if f_917_b == ""
	replace f_917_c = "Missing" if f_917_c == ""

	replace f_leader = "dc" if f_leader == "fc"
	replace f_leader = "jm" if f_leader == "im"
	replace f_leader = "pc" if f_leader == "  " & f_917_b == "Missing"
	replace f_leader = "cc" if f_917_c == "PA" & f_917_b == "D Dramatic work and accompanying music" & f_leader == "  " 
	replace f_leader = "gm" if f_917_c == "PA" & f_917_b == "X Motion picture or filmstrip" & f_leader == "  " 
	replace f_leader = "am" if f_917_c == "TX" & f_917_b == "B Non-dramatic literary work" & f_leader == "  " 

	replace f_917_b = "6 Serial" if f_917_c == "Vol. 18, no. 2, 1993."
	replace f_917_c = "TX" if f_917_c == "Vol. 18, no. 2, 1993."
	replace f_917_c = "TX" if f_917_c == "6 Serial"
	replace f_917_c = "TX" if f_917_c == "CA"
	drop if f_027 == "Adm"

	replace f_917_b = "B Non-dramatic literary work" if f_917_b == "B Non-dramatic literary Work" | f_917_b == "B Non-dramatic literary works" | f_917_b == "B" | f_917_b == "Bl"
	replace f_917_b = "N Sound recording and music" if f_917_b == "N  Sound recording and music"
	replace f_917_b = "Q Multimedia kit" if f_917_b == "Q Multimedia Kit"
	replace f_917_b = "6 Serial" if f_leader == "as"
	replace f_917_b = "B Non-dramatic literary work" if f_leader == "am" & f_917_b == "Missing" & (f_017 == "TX" | f_017 == "TXu")
	replace f_917_b = "B Non-dramatic literary work" if f_leader == "am" & f_917_b == "Missing" & (f_017 == "PA" | f_017 == "PAu")
	replace f_917_b = "B Non-dramatic literary work" if f_leader == "am" & f_917_b == "Missing" & (f_017 == "VA" | f_017 == "VAu")
	replace f_917_b = "D Dramatic work and accompanying music" if f_leader == "cc" & f_917_b == "Missing" 
	replace f_917_b = "U Sound recording" if f_leader == "cm" & f_917_b == "Missing" & (f_017 == "SR" | f_017 == "SRu")
	replace f_917_b = "M Musical work" if f_leader == "cm" & f_917_b == "Missing" & (f_017 == "PA" | f_017 == "TX")
	replace f_917_b = "N Sound recording and music" if f_leader == "dc" & f_917_b == "Missing" 
	replace f_917_b = "X Motion picture or filmstrip" if f_leader == "gm" & f_917_b == "Missing" 
	replace f_917_b = "A Sound recording and non-dramatic text" if f_leader == "ic" & f_917_b == "Missing" 
	replace f_917_b = "U Sound recording" if f_leader == "km" & f_917_b == "Missing" & (f_017 == "SR" | f_017 == "SRu")
	replace f_917_b = "S Art work" if f_leader == "km" & f_917_b == "Missing" & (f_017 == "VA" | f_017 == "VAu")
	replace f_917_b = "C Machine-readable work or computer program" if row_id == 609166 & file == 19
	replace f_917_b = "C Machine-readable work or computer program" if f_leader == "mm" & f_917_b == "Missing" 
	replace f_917_b = "Unspecified preregistration" if f_leader == "oc" & f_917_b == "Missing" 

	gen rec_type = "recordation" if f_027 == "V"
	replace rec_type = "recordation" if f_leader == "pc" 
	replace rec_type = "recordation" if f_leader == "pc"
	replace rec_type = "renewal" if f_027 == "RE" | f_017 == "RE"
	replace rec_type = "cancellation" if f_017 == "CAN"
	replace rec_type = "preregistration" if f_027 == "PRE" | f_017 == "PRE" | f_leader == "oc"
	replace rec_type = "cancellation" if f_leader == "tc"
	replace rec_type = "registration" if rec_type == ""

	gen work_type = "na" if rec_type == "cancellation" | rec_type == "recordation"
	replace work_type = f_917_b if work_type != "na"
	drop f_leader f_017 f_027 f_917_b f_917_c fr
	
	frame change hold
	frameappend default
	frame change default
}
*****
frame change hold
replace work_type = rec_type if rec_type != "registration"

replace work_type = "Architectural_drawing" if work_type == "1 Architectural drawing"
replace work_type = "Architectural_model" if work_type == "3 Architectural model"
replace work_type = "Serial" if work_type == "6 Serial"
replace work_type = "Sound_recording_and_non_dramatic_text" if work_type == "A Sound recording and non-dramatic text"
replace work_type = "Non_dramatic_literary_work" if work_type == "B Non-dramatic literary work"
replace work_type = "Machine_readable_work_or_computer_pro" if work_type == "C Machine-readable work or computer program"
replace work_type = "Dramatic_work_and_accompanying_music" if work_type == "D Dramatic work and accompanying music"
replace work_type = "Sound_recording_and_drama" if work_type == "E Sound recording and drama"
replace work_type = "Cartographic_work" if work_type == "F Cartographic work"
replace work_type = "Musical_work" if work_type == "M Musical work"
replace work_type = "Sound_recording_and_music" if work_type == "N Sound recording and music"
replace work_type = "Multimedia_kit" if work_type == "Q Multimedia kit"
replace work_type = "Art_work" if work_type == "S Art work"
replace work_type = "Sound_recording" if work_type == "U Sound recording"
replace work_type = "Motion_picture_or_filmstrip" if work_type == "X Motion picture or filmstrip"
replace work_type = "Mask_work" if work_type == "Z Mask work"
replace work_type = "Cancellation" if work_type == "cancellation"
replace work_type = "Preregistration" if work_type == "preregistration"
replace work_type = "Recordation" if work_type == "recordation"
replace work_type = "Renewal" if work_type == "renewal"
replace work_type = "Toy_or_game" if work_type == "G Toy or game"
replace work_type = "Small_household_item" if work_type == "H Small household item"
replace work_type = "Technical_drawing_or_model" if work_type == "I Technical drawing or model"
replace work_type = "Jewelry" if work_type == "J Jewelry"
replace work_type = "Two-dimensional_art_work" if work_type == "K Two-dimensional art work"
replace work_type = "Print_or_label" if work_type == "L Print or label"
replace work_type = "Original_work_of_art" if work_type == "O Original work of art"
replace work_type = "Photos_or_slides" if work_type == "P Photos or slides"
replace work_type = "Textile" if work_type == "T Textile"
replace work_type = "Choreography_or_pantomime" if work_type == "Y Choreography or pantomime"
replace work_type = "Art_reproduction" if work_type == "R Art reproduction"
frame drop default
frame rename hold default
frame change default

forvalues i == $start_fn/$end_fn {
	dis `i'
	capture frame drop temp_frame
	frame put if file == `i' , into(temp_frame) 
	frame change temp_frame
	save "${DIR}\temp\Index\index per file\File `i'", replace
	frame change default 
	frame drop temp_frame
}

forvalues i == $start_fn/$end_fn {
	dis `i'
	use "${DIR}\temp\parsed uncat data\usco_public_catalog_data_file`i'_ln_data.dta", clear
	merge m:1 row_id using "${DIR}\temp\Index\index per file\File `i'"
	drop _merge
	save "${DIR}\temp\Long Data\with cat\File `i'", replace
}

global names = "Architectural_drawing	Architectural_model	Serial	Sound_recording_and_non_dramatic_text	Non_dramatic_literary_work	Machine_readable_work_or_computer_pro	Dramatic_work_and_accompanying_music	Sound_recording_and_drama	Cartographic_work	Toy_or_game	Small_household_item	Technical_drawing_or_model	Jewelry	Two_dimensional_art_work	Print_or_label	Musical_work	Sound_recording_and_music	Original_work_of_art	Photos_or_slides	Multimedia_kit	Art_reproduction	Art_work	Textile	Sound_recording	Preregistration	Cancellation	Motion_picture_or_filmstrip	Choreography_or_pantomime	Mask_work	Recordation Renewal"

forvalues j = 1/31 {	
	qui: clear 
	qui: global w = word("$names", `j')
	dis "$w"
	qui: global fn = word("$names",`j')
	if "$fn" != "Preregistration" &	"$fn" != "Cancellation" &	"$fn" != "Recordation" &	"$fn" != "Renewal" {
		global pref = "REG_"
	}
	else {
		global pref = ""
	}
	qui: save "${DIR}/${output_folder}/${pref}${fn}", emptyok replace

	forvalues i == $start_fn/$end_fn {
		dis "$w `i'"
		qui: use "${DIR}\temp\Long Data\with cat/File `i'.dta", clear
		qui: keep if work_type == "$fn"
		replace var_nm = hold_var_nm
		qui: keep row_id v_ var_nm
		qui: append using "${DIR}/${output_folder}/${pref}${fn}"
		qui: save "${DIR}/${output_folder}/${pref}${fn}", replace
	}
}		