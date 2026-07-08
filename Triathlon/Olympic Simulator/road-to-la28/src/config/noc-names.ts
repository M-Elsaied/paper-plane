/**
 * NOC code -> country name, for the nations that appear in triathlon rankings
 * plus the major emerging federations. Unknown NOCs fall back to the raw code
 * (handled by `nationName`). This is a tunable default — extend as new nations
 * enter the rankings; nothing breaks if a code is missing.
 */

const NAMES: Record<string, string> = {
  // Europe
  GBR: "Great Britain", FRA: "France", GER: "Germany", ESP: "Spain", ITA: "Italy",
  POR: "Portugal", NED: "Netherlands", BEL: "Belgium", SUI: "Switzerland", AUT: "Austria",
  NOR: "Norway", SWE: "Sweden", DEN: "Denmark", FIN: "Finland", IRL: "Ireland",
  HUN: "Hungary", CZE: "Czechia", SVK: "Slovakia", POL: "Poland", ROU: "Romania",
  CRO: "Croatia", SLO: "Slovenia", SRB: "Serbia", UKR: "Ukraine", EST: "Estonia",
  LAT: "Latvia", LTU: "Lithuania", LUX: "Luxembourg", MON: "Monaco", GRE: "Greece",
  BUL: "Bulgaria", TUR: "Türkiye", ISR: "Israel", ISL: "Iceland", GEO: "Georgia",
  // Americas
  USA: "United States", CAN: "Canada", MEX: "Mexico", BRA: "Brazil", ARG: "Argentina",
  CHI: "Chile", COL: "Colombia", ECU: "Ecuador", PER: "Peru", VEN: "Venezuela",
  URU: "Uruguay", CRC: "Costa Rica", GUA: "Guatemala", PUR: "Puerto Rico", BER: "Bermuda",
  DOM: "Dominican Republic", BAR: "Barbados", JAM: "Jamaica", ISV: "US Virgin Islands",
  // Asia
  JPN: "Japan", KOR: "South Korea", CHN: "China", TPE: "Chinese Taipei", HKG: "Hong Kong",
  KAZ: "Kazakhstan", UZB: "Uzbekistan", IND: "India", THA: "Thailand", PHI: "Philippines",
  INA: "Indonesia", SGP: "Singapore", MAS: "Malaysia", VIE: "Vietnam", IRI: "Iran",
  UAE: "United Arab Emirates", QAT: "Qatar", KSA: "Saudi Arabia", JOR: "Jordan", LBN: "Lebanon",
  // Africa
  RSA: "South Africa", EGY: "Egypt", MAR: "Morocco", TUN: "Tunisia", ALG: "Algeria",
  KEN: "Kenya", NGR: "Nigeria", NGA: "Nigeria", ZIM: "Zimbabwe", NAM: "Namibia",
  MRI: "Mauritius", SEY: "Seychelles", BOT: "Botswana", UGA: "Uganda", CPV: "Cape Verde",
  // Oceania
  AUS: "Australia", NZL: "New Zealand", FIJ: "Fiji", PNG: "Papua New Guinea", SAM: "Samoa",
  COK: "Cook Islands", GUM: "Guam",
};

/** Full country name for a NOC code, falling back to the code itself. */
export function nationName(noc: string): string {
  return NAMES[noc] ?? noc;
}
