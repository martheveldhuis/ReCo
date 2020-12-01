## Notes

Sinds vorige week:
- Beter een idee van wat de mensen van BIS willen weten
- Doel: metrics?

Code Jennifer
- Foutjes in feature generator en in het uitpakken v/h model
- python 2.7 = echt deprecated
- daardoor andere functionaliteiten ook deprecated (oude versies)

Wat ik heb gedaan is:
1. In de oude python 2.7 versie het model uitgepakt (dat op een deprecated manier is gegenereerd)
2. Met geupdate joblib het weer ingepakt (nog steeds in python 2.7)
3. Deze vervolgens in colab notebook uitgepakt met nieuwere python (3.6) en pickle
  - Dit gaf wel wat warnings, maar o.b.v. output v/h model lijkt het wel te werken.

Dus?
- Hertrain het model in iig python 3.6. en maak daar nieuwe pickles van zodat het blijft werken!
- Gebruik mijn nieuwe pickle v/h model.

Aanstaande week:
- what-if tool van google laden om zo wat explanations te genereren
- lezen + notities:
  - focus op contrastive explanations bij NOC
  - focus op features
  - terug in mijn survey kijken naar mogelijke geschikte XAI methodes
  - metrics for XAI
- proposal te starten
