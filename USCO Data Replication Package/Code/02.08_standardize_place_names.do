********************************************************************************
* Standardize place names                                                      *
* Last modified April 10, 2024                                                 *
********************************************************************************

global list = ""
capture confirm variable place_of_1st_pub
if _rc == 0 {
	global list = `""place_of_1st_pub""'
}
forvalues i = 1/10 {
	capture confirm variable author_`i'_domicile
	if _rc == 0 {
		global list = `"$list"' + `" "' + `""author_`i'_domicile""'
	}
	capture confirm variable author_`i'_citizanship
	if _rc == 0 {
		global list = `"$list"' + `" "' + `""author_`i'_citizanship""'
	}
}
dis `"$list"'
foreach k in $list  {
	capture confirm variable `k'
	if _rc == 0 {
		capture drop temp
		gen v_ = trim(lower(`k'))
		capture drop country
		gen country = ""
		global cnt = `""afghanistan" "albania" "algeria" "american samoa" "andorra" "angola" "anguilla" "antarctica" "antigua" "barbuda" "argentina" "armenia" "aruba" "australia" "austria" "azerbaijan" "bahamas" "bahrain" "bangladesh" "barbados" "belarus" "belgium" "belize" "benin" "bermuda" "bhutan" "bolivia" "bosnia" "herzegovina" "botswana" "brazil" "brunei" "bulgaria" "burkina faso" "burundi" "cambodia" "cameroon" "canada" "cape verde" "cayman islands" "central african republic" "chad" "chile" "china" "christmas island" "colombia" "comoros" "congo" "cook islands" "costa rica" "ivory coast" "croatia" "cuba" "cyprus" "czech republic" "denmark" "djibouti" "dominica" "dominican republic" "east timor" "ecuador" "egypt" "el salvador" "equatorial guinea" "england" "eritrea" "estonia" "ethiopia" "falkland islands" "faroe islands" "fiji" "finland" "france" "french guiana" "french polynesia" "french southern territories" "gabon" "gambia" "georgia" "germany" "ghana" "gibraltar" "greece""'

		foreach i in $cnt {
			gen temp1 = strpos(v_, "`i'") if country == ""
			replace country = "`i'" if temp1 > 0 & country == ""
			dis "`i'"
			drop temp1
		}

		global cnt = `""greenland" "grenada" "guadeloupe" "guatemala" "guinea" "guinea-bissau" "guyana" "haiti" "holy see" "honduras" "hong kong" "hungary" "iceland" "india" "indonesia" "iran" "iraq" "ireland" "israel" "italy" "ivory coast" "jamaica" "japan" "jordan" "kazakhstan" "kenya" "kiribati" "korea" "kosovo" "kuwait" "kyrgyzstan" "latvia" "lebanon" "lesotho" "liberia" "libya" "liechtenstein" "lithuania" "luxembourg" "macau" "madagascar" "malawi" "malaysia" "maldives" "mali" "malta" "marshall islands" "martinique" "mauritania" "mauritius" "mayotte" "mexico" "micronesia" "moldova" "monaco" "mongolia" "montenegro" "montserrat" "morocco" "mozambique" "myanmar" "burma" "namibia" "nauru" "nepal" "netherlands" "netherlands antilles" "new caledonia" "new zealand" "nicaragua" "niger" "nigeria" "niue" "north macedonia" "northern mariana islands" "norway" "oman" "pakistan" "palau" "palestinian territories" "panama""'

		foreach i in $cnt {
			gen temp1 = strpos(v_, "`i'") if country == ""
			replace country = "`i'" if temp1 > 0 & country == ""
			dis "`i'"
			drop temp1
		}

		global cnt =  `""papua new guinea" "paraguay" "peru" "philippines" "pitcairn" "poland" "portugal" "qatar" "reunion island" "romania" "russia" "rwanda" "saint kitts and nevis" "saint lucia" "saint vincent" "grenadines" "samoa" "san marino" "sao tome and principe" "saudi arabia" "senegal" "serbia" "seychelles" "sierra leone" "singapore" "slovakia" "slovenia" "solomon islands" "somalia" "south africa" "south sudan" "spain" "sri lanka" "sudan" "suriname" "swaziland" "sweden" "switzerland" "syria" "taiwan" "tajikistan" "tanzania" "thailand" "tibet" "timor" "togo" "tokelau" "tonga" "trinidad" "tobago" "tunisia" "turkey" "turkmenistan" "turks and caicos" "tuvalu" "uganda" "ukraine" "united arab emirates" "united kingdom" "uk" "united states" "uruguay" "uzbekistan" "vanuatu" "vatican" "venezuela" "vietnam" "virgin islands"  "wallis and futuna" "western sahara" "yemen" "zambia" "zimbabwe""'

		foreach i in $cnt {
			gen temp1 = strpos(v_, "`i'") if country == ""
			replace country = "`i'" if temp1 > 0 & country == "" 
			dis "`i'"
			drop temp1
		}

		global cnt =  `""alabama"	"arizona"	"arkansas"	"california"	"colorado"	"connecticut"	"delaware"	"district of columbia"	"florida"	"guam"	"hawaii"	"illinois"	"iowa"	"kansas"	"kentucky"	"louisiana"	"maine"	"maryland"	"massachusetts"	"michigan"	"minnesota"	"mississippi"	"missouri"	"montana"	"nebraska"	"nevada"	"new hampshire"	"new jersey"	"new york"	"north carolina"	"north dakota"	"ohio"	"oklahoma"	"ontario"	"oregon" "pennsylvania"	"puerto rico"	"rhode island"	"south carolina"	"tennessee"	"texas"	"u.s. carribean islands"	"u.s. minor outlying"	"u.s. misc. pacific"	"usa"	"utah"	"vermont"	"virginia"	"washington"	"west virginia"	"wisconsin" "alaska" "idaho" "tennesee" "wyoming" "south dakota""'

		foreach i in $cnt {
			gen temp1 = strpos(v_, "`i'") if country == ""
			replace country = "united states" if temp1 > 0 & country == "" 
			dis "`i'"
			drop temp1
		}

		global cnt =  `""british columbia"	"new brunswick"	"newfoundland and labrador"	"nova scotia"	"prince edward island"	"quebec" "alberta" "northwest territories" "saskatchewan""'

		foreach i in $cnt {
			gen temp1 = strpos(v_, "`i'") if country == ""
			replace country = "canada" if temp1 > 0 & country == "" 
			dis "`i'"
			drop temp1
		}

		replace v_ = "not known" if v_ == "no place, unknown, undetermine" | v_ == "various places"
		replace v_ = "macedonia" if v_ == "macedonia (short form)" | v_ == "macedonia"
		replace v_ = "" if length(v_) < 4
		replace country = "canada" if regexm(v_ , "ottawa") & country == "" 
		replace country = "australia" if regexm(v_ , "sydney") &  country == "" 
		replace country = "canada" if regexm(v_ , "calgary") & country == "" 
		replace country = "canada" if regexm(v_ , "vancouver") &  country == "" 
		replace country = "canada" if regexm(v_ , "toronto") & country == "" 
		replace country = "uk" if regexm(v_ , "london") & country == "" 
		replace country = "switzerland" if regexm(v_ , "zurich") &  country == "" 
		replace country = "japan" if regexm(v_ , "osaka") & country == "" 
		replace country = v_ if country == ""
		replace `k' = country
		drop country v_
	}

}