## Notities

Francisca Duijs van BIS

#### Interpretatie pt1

Publiek = DNA deskundigen van de afdeling BIologische Sporen (BIS).
Zij hebben dus een biologische achtergrond, en maken gebruik van het profiel, aangeleverde informatie, en bepaalde ondersteunende software, om conclusies te trekken. Zij rapporteren dit, en moet kunnen worden verdedigd. Zo kan bewijs voor een zaak worden aangeleverd.

QOL-00711 document beschrijft stappen die de experts doorlopen om de profielen te analyseren. Bestaat uit 6 stappen, eerste 3 vandaag.

1. Bepalen enkelvoudig / mengsel
2. Bepalen hoeveel donoren
3. Bepalen welke mengverhoudingen

Nu wordt stap 1 en 2 gedaan o.b.v.
- MAC (wanneer MAC 2 is, wordt er uitgegaan van 1 contributor, etc.)
- Informatie over de kwaliteit van het DNA
  - Degradatie (breuken in het DNA)
  - Lage hoeveelheid DNA geeft lagere kwaliteit (weinig DNA geeft meer stutter -> stutter filters kunnen pieken van contributors weghalen -> onderschatting van NOC)
  - Hoeveelheid (gerelateerde) contributors geeft lagere kwaliteit
- NOC tool die alleen een NOC nummer, en een probability geeft

Voor stap 3 worden de hoogtes van de pieken verder geanalyseerd, maar daarvoor moeten dus de NOC bekend zijn. Zo worden profielen deconvolved. Met name met als doel om de hoofd donor(en) te achterhalen.

**Mijn doel ligt dus in stap 2: geef de NOC tool een uitleg o.b.v. de features waar de experts normaal ook naar kijken. Dus bijv. hoogtes van pieken / MAC / TAC / percentage pieken onder een threshold (hangt samen met kwaliteit) / etc.**

Een hoop processing is al gedaan op de data die ik heb (thresholds / filters / handmatig). Er worden geen geslacht-specifieke markers gebruikt omdat de data die ik heb, alleen mannen betreft.

#### Interpretatie pt2

Begin met aantal donoren
1-> bereken bewijskracht
2-4 -> aantal mismatches met referenties
als teveel -> niet rekenen
anders, bereken bewijskracht (LR)

Mismatches = onverklaarde alleles, dropouts in vergelijking met referentie.

LR=1 neuraal
LR>1 H1 true
LR<1 H2 true

quantitative model
- genotypic probabilities (hardy weinberg equilibrium)
- peak height information
- drop-out/-in based on peak height models

LR = (Hd:V+POI)/(Hp:V+U)
PR(evidence|Hd) = 1 wanneer beide allelen van victim en person of interest verklaard zijn

drop-out en -in.
met een extra parameter d (probability of allele dropping out)

extra optimizer voor mixture proportions, peak height expectation (gamma distributie), peak height variance, degradatie etc.

bij NFI EuroForMix -> DNAStatistX

model rekent alle probability combinaties (allele combis prob + drop-out prob + drop-in prob)

kans drop-out is per allelen

voor drop-in met exponentiele distributie. hoge piekhogte -> probably not drop-in

er wordt ook rekening gehouden met degradatie voor verwachting van de piekhoogte o.b.v. grootte.

Laatste stap is validatie: vergelijking tussen verwacht en geobserveerd

LR van 10.000 is te rapporteren.



#### Verklaringen
De mensen die gebruik maken van de software zijn als het goed is bekend met alle parameters, e.g. invloed op uitkomst etc., maar grafieken en dergelijke helpen wel.

Wanneer het NOC wordt onderschat, heb je veel onverklaarde pieken en is het resultaat niet te valideren (dus de vergelijking tussen verwacht en geobserveerd houdt niet stand). Maar het is handiger om in eerste instantie uit te gaan van minder donoren, want:

* Het is lastig om te verklaren in een rechtszaak als je een extra contributor gebruikt om ruis te verklaren. Dan krijg je vragen bij bijv. een zedenzaak (waar je 2 mensen verwacht) "Waarom heb je 3 mensen in je model gestopt?".
* Met meer NOC duurt de berekening langer
  * 3 of minder onbekende donoren duurt < 1 minuut
  * 4 onbekende donoren duurt 15 minuten - 3.5 uur o.b.v. MAC tussen 6-10.
* Met meer NOC wordt de mengbijdrage heel laag van een contributor die er niet is.
* Met meer NOC mag je sneller niet rekenen naar de LR door teveel aan mismatches.

Voor user study is het handig om te vragen om welke specifieke profielen vooral van interesse zijn, die een extra verklaring nodig hebben omdat bijvoorbeeld de MAC niet aansluit met wat er uit het NOC model komt.
