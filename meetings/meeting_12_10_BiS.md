## Update BiS 2020/12/10

#### Data + model

Dataset is oké nu.

Regressie doet het iets beter (mogelijk door relatie 1-2-3-4-5). Maar in de praktijk zal je bij resultaat 4.3 toch tussen 4 en 5 gaan kijken. Dat is hetzelfde als je een classificatie resultaat van 4:50% en 5:50% hebt.

#### Explanations

Corina lijkt de meerwaarde in te zien van counterfactuals. Normaal kijken de experts naar de MAC, TAC, piekhoogtes, etc. en maken ze een inschatting. Komt dit overeen met het model, kijken ze niet verder. Als het niet overeenkomt, willen ze weten waarop het model dit baseert. Daarbij moet een expert vooral de keuze maken tussen 2 mogelijke uitkomsten.

- Globale explanations vertellen iets over de algemene relatie van feature waardes met de uitkomst, maar men wil uiteindelijk informatie krijgen over individuele voorspellingen.
- SHAP geeft feature waardes die het meest hebben bijgedragen aan de voorspelling, maar niet waarom het niet de andere uitkomst is.

Als counterfactuals worden gebruikt, e.g. "stel waarde x was 0 ipv 1, dan werd een andere voorspelling gemaakt", dan weet de expert:
1. Dat feature x belangrijk was om tussen de 2 predictions te beslissen.
2. Dat waarde x = 1 meer past bij de originele prediction.
3. Als feature x = 1 niet past bij de originele prediction, de classifier waarschijnlijk een fout maakt.

Bij het genereren van counterfactuals moet je opletten dat de feature values die je genereert binnen de training distributie vallen. Daarmee zorg je dat er geen onmogelijke feature combinaties maakt. Hier moet je zeker op letten, en dat is iets wat ik wil meenemen in mijn evaluatie.

*Kunnen we counterfactuals compleet kunnen genereren ipv feature waardes aanpassen?* Je wil juist weten hoe een vergelijkbaar profiel (met dus 2-3 aanpassingen) anders zou worden geclassificeerd.

#### User studies

- Leg het eerst voor bij Corina voor input.
- Gebruik vooral foute predictions / twijfelgevallen.
- Neem later contact op met de mensen met een concreet plan (tijdsduur / taken / wat gebeurt met resultaten)


----------------------------------------------

We hebben hier een erg specifieke context:
- Weinig data
- Veel features
- Veel afhankelijke features

Dus misschien interessant om juist onderzoek te doen naar hoe counterfactuals voor deze situatie goed kunnen worden geïmplementeerd.

Er is een generatief model voor het genereren van profielen (e.g. origineel profiel + 1 piek). Kunnen we dit gebruiken voor het maken van counterfactuals?

Ook interessant:  
why 4 and not 5.  
why 5 and not 4.  
