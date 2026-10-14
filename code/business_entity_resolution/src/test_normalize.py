"""Unit tests on real records taken from the dataset.  Run:  python src/test_normalize.py"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_cleaning import NameNormalizer, NameResources, normalize_address  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
res = pickle.load(open(os.path.join(HERE, "..", "artifacts", "resources.pkl"), "rb"))
NN = NameNormalizer(NameResources(res["vocab"], res["native_lex"], res["eng_phon"]))

NAME_CASES = [
    # (raw, country, expected subset of output fields)
    ("Smith's American Grading P.C.", "US", {"name_core": "smiths american grading", "name_legal": "pc"}),
    ("Smith's American American Grading Inc", "US", {"name_core": "smiths american grading", "name_legal": "inc"}),
    ("Smith's P.C. Grading American", "US", {"name_core": "smiths grading american", "name_legal": "pc"}),
    ("Nexrizaonyx", "US", {"name_type": "brand", "name_brand": "nexrizaonyx"}),
    ("Verayuma One", "US", {"name_type": "brand"}),
    ("Delta Tetlecommunication Inc", "US", {"name_type": "normal", "name_core": "delta tetlecommunication"}),
    ("Halolyra DBA Sauvegarde Amicale SAS", "France", {"name_type": "alias", "name_core": "sauvegarde amicale",
                                                       "name_legal": "sas", "name_brand": "halolyra"}),
    ("Veosyn formerly: HB Fargo, LLC", "US", {"name_type": "alias", "name_core": "hb fargo", "name_legal": "llc"}),
    ("Brixkor Labs t/a Inter Élémentaire SARL", "France", {"name_type": "alias", "name_core": "inter elementaire",
                                                            "name_legal": "sarl"}),
    ("Deleon's Select Pizza", "US", {"name_core": "deleons select pizza"}),
    ("Belocalo Co fka Deleon's Select Pizza", "US", {"name_core": "deleons select pizza", "name_type": "alias"}),
    ("sterlingexpressjersey.com", "US", {"name_type": "domain", "name_core": "sterling express jersey"}),
    ("ruthespinnacletax.com", "US", {"name_type": "domain", "name_concat": "ruthespinnacletax"}),
    ("jspacsphere.c0m (ID: 56210)", "US", {"name_type": "domain", "name_concat": "jspacsphere"}),
    ("@robustworks", "India", {"name_type": "handle", "name_core": "robust works"}),
    ("M/s silverlightbrothers.com", "India", {"name_type": "domain", "name_core": "silverlight brothers"}),
    ("Ridge LLC Service | www.ridgellc.c0m", "US", {"name_core": "ridge service", "name_legal": "llc"}),
    ("MW Sp0rts Private (Limited)", "India", {"name_core": "mw sports", "name_legal": "ltd pvt"}),
    ("Jp Institute of Techno1ogy Care Ltd", "India", {"name_core": "jp institute technology care", "name_legal": "ltd"}),
    ("Advita Limited Ass0ciates &", "India", {"name_core": "advita associates", "name_legal": "ltd"}),
    ("Aryans (lndia) Pvt Ltd Center", "India", {"name_core": "aryans india center", "name_legal": "ltd pvt"}),
    (">> Aryans (lndia) Pvt Ltd Center", "India", {"name_core": "aryans india center"}),
    ("*** SERVICES-SIPCO PAVERS PRIVATE", "India", {"name_core": "services sipco pavers", "name_legal": "pvt"}),
    ("Pvt. EFS Print Ventures Ltd.", "India", {"name_core": "efs print ventures", "name_legal": "ltd pvt"}),
    ("Hayat Realtors Limited (Private)", "India", {"name_core": "hayat realtors", "name_legal": "ltd pvt"}),
    ("Mr Red Academy Ventures", "India", {"name_core": "red academy ventures"}),
    ("Shree Software Co.", "India", {"name_core": "shree software", "name_legal": "co"}),
    ("Sree Services Limited", "India", {"name_core": "shree services", "name_legal": "ltd"}),
    ("Dr pioneerpackers.com", "India", {"name_type": "domain", "name_core": "pioneer packers"}),
    ("Ziya-Charitable #26622 Trust", "India", {"name_core": "ziya charitable trust"}),
    ("Nylanex - 8984140124", "India", {"name_type": "brand", "name_core": "nylanex"}),
    ("Lumgilddrex #89752", "India", {"name_type": "brand"}),
    ("Private Jupiter & Brtsehrs Group Limited", "India", {"name_core": "jupiter brtsehrs group", "name_legal": "ltd pvt"}),
    ("Pirvate Sahni Systems Ltd", "India", {"name_core": "sahni systems", "name_legal": "ltd pvt"}),
    ("WV-Private Limited Partners (ID: 67437)", "India", {"name_core": "wv partners", "name_legal": "ltd pvt"}),
    ("Prime Realty Ventures Public Limited", "India", {"name_core": "prime realty ventures", "name_legal": "ltd public"}),
    ("Gosselin, Lexis, M.D.", "US", {"name_core": "gosselin lexis", "name_legal": "md"}),
    ("Darrelle Sullivan, DMD, DDS PC", "US", {"name_core": "darrelle sullivan", "name_legal": "dds dmd pc"}),
    ("A Integrated Kongsberg Inc.", "US", {"name_core": "integrated kongsberg", "name_legal": "inc"}),
    ("Rahn Clean Beauty LLC", "US", {"name_core": "rahn clean beauty", "name_legal": "llc"}),
    ("RAHN CLÉAN BEAUTY LLC", "US", {"name_core": "rahn clean beauty", "name_legal": "llc"}),
    ("Vargas Pioneer", "US", {"name_core": "vargas pioneer"}),
    ("#várgaspi0neer", "US", {"name_type": "handle", "name_core": "vargas pioneer"}),
    ("IMC", "US", {"name_type": "acronym", "name_core": "imc"}),
    ("Interstate Medical Center", "US", {"name_acronym": "imc"}),
    ("Oswald and Carswell 5ervice", "US", {"name_core": "oswald carswell service"}),
    ("Metro Cascade Ctiroup LLC", "US", {"name_core": "metro cascade ctiroup"}),
    ("\"\"\"maison Collectif Participations SAS\"", "France", {"name_core": "maison collectif participations",
                                                              "name_legal": "sas"}),
    ("Maison de Santé Sainte André", "France", {"name_core": "maison sante sainte andre"}),
    ("Lycée de l'Orne", "France", {"name_core": "lycee orne"}),
    ("Chambre Medico SARL", "France", {"name_core": "chambre medico", "name_legal": "sarl"}),
    ("Memoire (France) Sportive S.A.", "France", {"name_core": "memoire france sportive", "name_legal": "sa"}),
    ("Tourcoing SARL  et Associés", "France", {"name_core": "tourcoing associes", "name_legal": "sarl"}),
    ("5arl Ecole Pique", "France", {"name_legal": "sarl"}),
    ("Lrm Management Limited", "India", {"name_core": "lrm management"}),
    ("Purva Sema [Samiti]", "India", {"name_core": "purva sema samiti"}),
    # native scripts
    ("ઇન્ડો કન્સલ્ટન્ટ્સ પ્રા. લિ.", "India", {"name_core": "indo consultants", "name_legal": "ltd pvt",
                                             "name_script": "gujarati"}),
    ("हॉटेल फाइनेंस प्राइवेट लिमिटेड", "India", {"name_legal": "ltd pvt"}),
    ("ஹோட்டல் ஃபைனான்ஸ் பிரைவேட் லிமிடெட்", "India", {"name_core": "hotel finance", "name_legal": "ltd pvt"}),
    ("तिरुपति ग्रीन प्रोड्यूसर प्राइवेट लिमिटेड", "India", {"name_legal": "ltd pvt"}),
    # short phonetic key (< 5 consonants): transliteration kept, matched through name_phon
    ("मोटर्स", "India", {"name_core": "motars", "name_phon": "mtrs"}),
    ("एजेंसीज", "India", {"name_core": "ejensij"}),                     # never 'essence'
    ("ऑटोमोबाइल्स", "India", {"name_core": "otomobails"}),               # independent candra-o kept
    ("ज्वेलर्स", "India", {"name_core": "jewellers"}),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "India", {"name_legal": "ltd pvt"}),
    ("एसएस इंटरनेशनल प्राइवेट लिमिटेड", "India", {"name_legal": "ltd pvt"}),
]

ADDR_CASES = [
    ("TX, 16901 Hayride Circle, Del Valle", "US",
     {"addr_words": "hayride cir del valle", "house_no": "16901", "city": "del valle", "state": "TX"}),
    ("016901 HAYRIDE CIR, DEL VALLE, TX", "US", {"addr_words": "hayride cir del valle", "house_no": "16901", "state": "TX"}),
    ("#16901 Hayride Cir, Del Valle, Texas", "US", {"addr_words": "hayride cir del valle", "house_no": "16901", "state": "TX"}),
    ("14929 Shady Grove Road, Unit K, Rockville, MD", "US", {"unit": "k", "house_no": "14929", "state": "MD",
                                                             "addr_words": "shady grv rd rockville"}),
    ("14929 Shady Groev Road, # K, Rockville, Maryland", "US", {"unit": "k", "state": "MD"}),
    ("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US", {"unit": "g", "addr_words": "cameron dr dundalk"}),
    ("IA, Iowa City, 1064 Newton Rd, Unit 11", "US", {"state": "IA", "city": "iowa city", "unit": "11", "house_no": "1064"}),
    ("NY, 125 Bowery, New York", "US", {"state": "NY", "city": "new york", "house_no": "125"}),
    ("635-639 243TH ST, DOUGLAS COUNTY, NE", "US", {"state": "NE", "addr_nums": "635 639 243", "house_no": "635"}),
    ("635 243rd Street, Douglas County, Nebraska", "US", {"state": "NE", "addr_nums": "635 243"}),
    ("15 SEWARD SAINT, WEST GLENS FALLS, NY", "US", {"addr_words": "seward st w glens falls"}),
    ("SHERWOOD CROSSING PLACE, null, MECHANICSVILLE, VA", "US", {"addr_words": "sherwood xing pl mechanicsville",
                                                                 "state": "VA"}),
    ("<NULL>, Texas, 02225 Poulin Avenue, Grand Prairie", "US", {"state": "TX", "house_no": "2225"}),
    ("9784  MOUNT KATAKA POINT, <NULL>, STERLING RANCH, CO", "US", {"state": "CO", "addr_words": "mt kataka pt sterling ranch"}),
    ("4817 Queens Chapel Terrace, Washington, DC", "US", {"state": "DC", "city": "washington"}),
    ("1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio", "US", {"state": "OH", "addr_nums": "1", "house_no": "1"}),
    ("Mrristown, PMB 1974, Tennessee, 445 Barkley Landing Dirve", "US", {"house_no": "445", "addr_nums": "445"}),
    ("205 River Drive, NC, Unit 1/2, Southport", "US", {"unit": "1/2", "addr_nums": "205", "city": "southport"}),
    ("9305 Cloverhurst Drive, Fl 0, Saint Louis, MO", "US", {"addr_nums": "9305",
                                                            "addr_words": "cloverhurst dr st louis"}),
    ("4308A FAULKNER COURT, PMB 5774, LINCOLN, NE", "US", {"house_no": "4308", "addr_nums": "4308"}),
    ("Door No 95 Frist Floor, Sec-17, Panchkula, Panchkula, HR", "India",
     {"house_no": "95", "addr_nums": "95 17", "addr_words": "frist sector panchkula"}),
    ("Building No 409 Div No 45 5Thcross Road Willingdon Island Cochin, Ernakulam, Kerala", "India",
     {"addr_nums": "409 45 5", "addr_words": "div cross rd willingdon island kochi ernakulam"}),
    ("Krishnabuilding2Ndfloor, West Bengal, Kolkata, Howrah, Roomno.207, 224A.J.C.Boseroad", "India",
     {"state": "WB"}),
    ("Rishi Towers, Box Colony 12, Hyderabad, Telangana", "India", {"addr_nums": "12"}),
    ("", "US", {"addr_missing": 1}),
    ("301, Om Shayona Arcade, Nr. Silver Oak University, S. G. Highway, Gota, Ahmedabad, Gujarat", "India",
     {"state": "GJ", "house_no": "301", "city": "ahmedabad",
      "addr_words": "om shayona arcade silver oak university s g hwy gota ahmedabad"}),
    ("ગુજરાત, Ahmedabad, #263 301, Ahmedabad", "India", {"state": "GJ", "addr_nums": "263 301", "city": "ahmedabad"}),
    ("H.NO 299 SNO 72/2/1, KAKA RESIDENCY, TAPKIR NAGAR, KALEWADI, PUNE, Maharashtra", "India",
     {"state": "MH", "house_no": "299"}),
    ("2Nd Floor N No241 O.No.85, 86 Rangarajapuram Main Rd, Chennai City Corporation, Chennai, Tamil Nadu", "India",
     {"state": "TN", "city": "chennai"}),
    ("NO 404 VILL CHODHA, SECTOR-22 NEXT TO ORIENTAL BLNK OF COMMERCE, NOIDA, उत्तर प्रदेश", "India",
     {"state": "UP", "house_no": "404"}),
    ("Tamil Nadu, SIEMA BUILDING, 8/4 RACE COURSE, NEAR THOMAS PARK, COIMBATORE, NO 001ST FLOOR, COIMBATORE", "India",
     {"state": "TN", "addr_nums": "8 4"}),
    ("A- 104 , FIRST FLOOR, GALI NO Â\x80\x93 18, NAGAR, MAYUR VIHAR PHASE -1, NEW DELHI, दिल्ली", "India",
     {"state": "DL", "city": "new delhi"}),
    ("Delhi, Gandhi Sadan, P K Road, New Delhi, 47, Delhi, New Delhi", "India", {"state": "DL", "city": "new delhi"}),
    ("1624 C WING RUNWAL PEARL, G.B. ROAD, THANE WEST, THANE, <NULL>, महाराष्ट्र", "India", {"state": "MH"}),
    ("Door No ##1, Near Girls, Govt. School, Gagsina, Karnal, null, HR", "India", {"state": "HR", "house_no": "1"}),
    ("C/O Pm Vijaykumar, No 11, Nn Pura 5Th Cross, City Post Banglore, Bangalore, Karnataka", "India",
     {"state": "KA", "house_no": "11", "city": "bangalore"}),
    ("Flat No. 601, Wing G, Radhika Park, S. No. Â\x80\x93 22, Wadgaonsheri, Pune, Maharashtra", "India",
     {"state": "MH", "house_no": "601"}),
    ("15, BOMBAY, MH", "India", {"addr_words": "mumbai", "state": "MH"}),
    ("5-3-1008 M S G Road, Hyderabad, Telangana", "India", {"state": "APTG"}),
    ("NO 5-3-1008 M S G ROD, HYDERABAD, Andhra Pradesh", "India", {"state": "APTG"}),
    ("12, Hyderabad, తెలంగాణ", "India", {"state": "APTG"}),
    ("17 Rue de Valmy, Lille, Hauts-de-France", "France", {"state": "HDF", "addr_words": "r valmy lille", "house_no": "17"}),
    ("17 R. DE VALMY, LILLE, Nord", "France", {"state": "HDF", "addr_words": "r valmy lille"}),
    ("N° 73 Avenue Marc Desbats, Pessac, Nouvelle-Aquitaine", "France", {"state": "NAQ", "house_no": "73",
                                                                         "addr_words": "av marc desbats pessac"}),
    ("Nº 23 CHEMIN DES GARENNES, ST-NAZAIRE, Pays de la Loire", "France", {"state": "PDL",
                                                                          "addr_words": "ch garennes st nazaire"}),
    ("15 bis Rue Brascassat, Bordeaux, Gironde", "France", {"state": "NAQ", "addr_words": "r brascassat bordeaux"}),
    ("La Teste-de-Buch, 15 Rue des Asphodeles, Nouvelle-Aquitaine", "France",
     {"city": "teste buch", "addr_words": "teste buch r asphodeles"}),
    ("12 RUE de lOrne, 44800, Saint-Herblain, Pays de la Loire", "France", {"addr_nums": "12 44800", "state": "PDL"}),
]


def run():
    fails = 0
    for raw, c, exp in NAME_CASES:
        out = NN.normalize(raw, c)
        bad = {k: (v, out[k]) for k, v in exp.items() if out[k] != v}
        if bad:
            fails += 1
            print(f"NAME FAIL {raw!r} [{c}]: " + "; ".join(f"{k}: want {w!r} got {g!r}" for k, (w, g) in bad.items()))
    for raw, c, exp in ADDR_CASES:
        out = normalize_address(raw, c)
        bad = {k: (v, out[k]) for k, v in exp.items() if out[k] != v}
        if bad:
            fails += 1
            print(f"ADDR FAIL {raw!r} [{c}]: " + "; ".join(f"{k}: want {w!r} got {g!r}" for k, (w, g) in bad.items()))
    total = len(NAME_CASES) + len(ADDR_CASES)
    print(f"{total - fails}/{total} cases passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
