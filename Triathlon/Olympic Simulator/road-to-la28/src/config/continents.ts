/**
 * NOC -> continent map, using World Triathlon's five continental confederations
 * (which drive the New Flag pathways). The Olympic quota reserves one New Flag
 * place per continent via Continental Games and one via the World Ranking, so an
 * athlete's continent decides who they compete against for those routes.
 *
 * Covers every NOC in the current rankings plus the major emerging triathlon
 * nations. Unknown NOCs resolve to null (handled gracefully by the engine).
 */

export type Continent = "Africa" | "Americas" | "Asia" | "Europe" | "Oceania";

export const CONTINENT_LABEL: Record<Continent, string> = {
  Africa: "Africa",
  Americas: "the Americas",
  Asia: "Asia",
  Europe: "Europe",
  Oceania: "Oceania",
};

/** New Flag World-Ranking priority order (places allocated in this sequence). */
export const NEWFLAG_PRIORITY: Continent[] = ["Africa", "Americas", "Asia", "Europe", "Oceania"];

const MAP: Record<string, Continent> = {};
const add = (c: Continent, nocs: string[]) => nocs.forEach((n) => (MAP[n] = c));

add("Europe", [
  "GBR", "FRA", "GER", "ESP", "ITA", "POR", "NED", "BEL", "SUI", "AUT", "NOR", "SWE",
  "DEN", "FIN", "IRL", "HUN", "CZE", "SVK", "POL", "ROU", "CRO", "SLO", "SRB", "UKR",
  "BLR", "EST", "LAT", "LTU", "LUX", "MON", "GRE", "BUL", "TUR", "ISR", "RUS", "AIN",
  "ALB", "AND", "MLT", "CYP", "ISL", "GEO", "ARM", "AZE", "MDA", "BIH", "MKD", "MNE",
]);
add("Americas", [
  "USA", "CAN", "MEX", "BRA", "ARG", "CHI", "COL", "ECU", "PER", "VEN", "URU", "PAR",
  "BOL", "CRC", "GUA", "ESA", "HON", "PAN", "DOM", "PUR", "BER", "TRI", "BAR", "BAH",
  "JAM", "ISV", "CAY", "AHO", "CUB", "NCA",
]);
add("Asia", [
  "JPN", "KOR", "CHN", "TPE", "HKG", "KAZ", "UZB", "IND", "THA", "PHI", "INA", "SGP",
  "MAS", "VIE", "IRI", "UAE", "QAT", "KSA", "JOR", "LBN", "SRI", "PAK", "BRN", "KUW",
  "OMA", "BHU", "NEP", "MGL", "PRK", "TJK", "KGZ", "TKM",
]);
add("Africa", [
  "RSA", "EGY", "MAR", "TUN", "ALG", "KEN", "NGR", "NGA", "ZIM", "NAM", "MRI", "SEY",
  "BOT", "UGA", "TAN", "GHA", "CIV", "CMR", "SEN", "ETH", "ZAM", "MAD", "CPV", "ANG",
]);
add("Oceania", [
  "AUS", "NZL", "FIJ", "PNG", "SAM", "TGA", "COK", "VAN", "GUM", "NRU", "SOL", "KIR",
]);

export function continentOf(noc: string | null | undefined): Continent | null {
  if (!noc) return null;
  return MAP[noc] ?? null;
}
