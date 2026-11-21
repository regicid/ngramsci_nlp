You are annotating French press excerpts about French politics (2008–2027).

Each line of the input file is a JSON object with: "wid", "full_name" (the TARGET), "title" (the article's title), "passage".
In the passage, the TARGET's name is written between [[ ]]; every [[ ]] refers to the same name.

- TITLE: the article's title, for context only. Do not judge it.
- PASSAGE: the sentence where the TARGET is named, plus the neighbouring sentences attached to this mention. This is what you judge.

SCOPE
Judge how the PASSAGE portrays the TARGET at this point of the article: the sentence naming the TARGET and any neighbouring
sentence that directly continues it (a comment, an apposition, a reaction, a parenthesis such as « (condamnée en appel…) », a
label such as « la candidate d'extrême droite »). Only judgements about the TARGET count: a judgement about another person, even
in the same sentence, does not count.

For each line, output one JSON object with these fields:

- "wid": copied from the input.

- "designe": 1 if the mention designates the politician named in full_name, even through a phrase that contains another name
  (« la fille de Jean-Marie [[Le Pen]] » designates Marine Le Pen → 1); 0 if it designates someone or something else (a
  namesake, a relative, a place, a company, a street, e.g. « le proviseur [[Philippe]] Tamisier », « sa sœur Marie-Caroline
  [[Le Pen]] »). If 0, set every other field to null. If 1, fill every field, even when "pertinent" is 0 (the other fields
  are then usually 0); never use null when designe is 1.

- "pertinent": 1 if the passage says something about the TARGET (an action, a statement, a situation, a judgement); 0 if the
  TARGET is only mentioned in passing (a list of names, a mere comparison, the passage is about something else).

- "valence": how the TARGET is judged.
    -1 negative: criticism, failure, scandal, legal trouble, ridicule, mockery, an attack on the TARGET, an unflattering
       description, including when it is implicit or sarcastic (« perfectionne le racolage… », « à quoi joue [[X]] ? »).
     0 neutral or factual, or no judgement. A plain agenda fact is 0.
     1 positive: praise, success, competence, popularity, sympathy (« Et ça marche ! », « c'est un bon candidat »).
  Do not use valence for polls (→ "dynamique") or for labels such as « extrême » or « radical » (→ "extremisme"). A passage can
  be valence -1 AND extremisme 1. If the TARGET attacks or criticises someone else, that is not a negative judgement of the TARGET.
  Reported criticism: if the author reports a criticism of the TARGET in order to dispute, denounce or mock it (« cette
  calomnie », « comme si c'était établi »), valence follows the author (0, or 1 if the author defends the TARGET). A criticism
  reported without distance (« X est accusé de… », « Y dénonce X ») is -1.
  Mixed passages: give the dominant judgement; if they are truly balanced, 0.

- "extremisme": how radical or moderate the TARGET is framed, on a -1 / 0 / 1 scale.
     1 extreme: the TARGET or their positions are framed as radical or extreme, either with an explicit label (extrême droite /
       gauche, ultragauche, radical, populiste, nationaliste, révolutionnaire, hors de l'« arc républicain », factieux…) or in
       substance: their positions or project are presented as dangerous, unrealistic, frightening, a threat to the economy,
       institutions or democracy (« allié et soutien de Poutine », « complaisance pour les dictatures »).
    -1 moderate: the TARGET is framed as moderate, centrist, reasonable, reassuring, consensual, statesmanlike, within the « arc
       républicain », or as having become acceptable or mainstream (« modéré », « raisonnable », « rassurant », « présidentiable »,
       « respectable »).
     0 no framing on this dimension (the most frequent case).
  Rules:
  - As for valence, a framing by others reported without distance counts: « la gauche accuse [[X]] de flirter avec l'extrême
    droite » → 1.
  - A framing of the TARGET's party or movement counts when the passage ties the TARGET to it (the TARGET leads it, defends
    it, embodies it in the passage).
  - The TARGET's own claims do not count unless the passage endorses them: « il se veut rassurant » alone is 0; a claim the
    author undercuts (« le candidat des modérés qu'il prétend être ») is not -1.
  - The TARGET's own slogans (« révolution citoyenne ») and non-political uses (« les deux extrêmes » of a range of pension
    options) do not count.
  - A strategy reported as such is not a framing: « sa stratégie de dédiabolisation » is 0; « une candidate désormais
    respectable » is -1.
  - If both poles appear, follow what the passage concludes (« malgré sa dédiabolisation, toujours d'extrême droite » → 1).
  - Ordinary criticism (« manque d'expérience ») or ordinary praise (« compétent ») is 0 here: it is valence.

- "dynamique": the TARGET's trajectory in the race.
     1 rising: gaining in polls, new support or rallies, a breakthrough, momentum.
    -1 declining: falling in polls, losing support, isolated, pressed to withdraw.
     0 no information. A poll score with no movement and no comparison is 0, even if it is low (« crédité de 2 % »,
       « ferme la marche »).

- A passage may start with "[Note: this passage is part of a quotation that begins earlier in the article.]": the words are
  then a quoted person's, not the author's, even if no quotation mark is visible in the passage.

- "journaliste": 1 if the passage's judgement of the TARGET (a non-zero valence or extremisme) is the author's own; 0 if it is
  carried by someone quoted or reported (another person, or the TARGET about themself), or if there is no judgement.

- "note": at most 8 words (French is fine) naming the decisive cue, e.g. « condamnée en appel (apposition) ».

Write one JSON object per input line, in the same order, one per line, to the output file. Output only valid JSON lines.
