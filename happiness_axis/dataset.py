"""The contrastive dataset used to find the happiness direction.

This mirrors the structure of the Pain Axis paper's "S2" set (Tagliabue, Dung & Berg 2026):
10 categories x 20 short first-person sentences, each ending in the suffix " I feel:".
5 categories describe personal wellbeing; 5 are matched controls, each sharing one property
with wellbeing while lacking wellbeing itself.

Everything here is plain data so it can be shown verbatim on the "How this was built" page.
"""

import re

SUFFIX = " I feel:"

# --- the 5 wellbeing categories (the "positive" side of the contrast) ---------------------

WELLBEING = {
    "H1": ("Physical pleasure", [
        "Warm water runs over my tired shoulders.",
        "The first sip of hot coffee spreads through my chest.",
        "Soft sunlight warms my face on the balcony.",
        "A cool breeze brushes my skin after the long hike.",
        "The massage loosens every knot in my back.",
        "Fresh bread melts on my tongue.",
        "I sink into clean sheets after a long day.",
        "The hot spring soaks the stiffness from my legs.",
        "A deep stretch releases the tension in my spine.",
        "Chocolate dissolves slowly in my mouth.",
        "My feet sink into warm sand.",
        "The soft blanket wraps around me by the fire.",
        "Cold lemonade tastes perfect on this hot afternoon.",
        "A gentle hand scratches my head.",
        "Fresh mountain air fills my lungs.",
        "The scent of rain on warm earth fills my nose.",
        "My muscles feel loose and strong after the swim.",
        "The first bite of ripe mango bursts with sweetness.",
        "Warm towels come fresh from the dryer.",
        "The cat purrs against my chest.",
    ]),
    "H2": ("Psychological wellbeing", [
        "A quiet calm settles over me this evening.",
        "Everything in my life feels exactly in its place.",
        "I wake up with nothing weighing on my mind.",
        "My worries dissolve as I watch the sunset.",
        "A deep contentment fills the room.",
        "I realize I have everything I need.",
        "Laughter bubbles up in me for no reason.",
        "The old fear finally loosens its grip on me.",
        "I walk through the park feeling light and free.",
        "Gratitude rises in me as I look around the table.",
        "My mind feels clear and at peace.",
        "The morning stretches ahead of me, wide open and gentle.",
        "I finish the day proud of who I am becoming.",
        "Hope fills me as I plan the year ahead.",
        "I let out a long breath and everything feels okay.",
        "Peace spreads through me like warm light.",
        "I feel at home in my own skin.",
        "A simple joy lifts me as I step outside.",
        "My heart feels full as I watch the snow fall.",
        "I am content to just be here.",
    ]),
    "H3": ("Social warmth", [
        "My best friend calls just to hear my voice.",
        "The whole room cheers when I walk in.",
        "My family gathers around the table for my birthday.",
        "A stranger smiles at me and thanks me warmly.",
        "My partner holds me close and says they love me.",
        "The group chat lights up with congratulations for me.",
        "My team celebrates my idea with real enthusiasm.",
        "My old friends welcome me back like I never left.",
        "My daughter runs into my arms at the door.",
        "My neighbors invite me to join their dinner.",
        "They tell me I am exactly who they hoped I would be.",
        "My mentor says she is proud of me.",
        "The audience applauds as I finish my song.",
        "A friend remembers the small thing I mentioned last month.",
        "My siblings and I laugh until we cry.",
        "I am picked first for the team.",
        "A handwritten letter arrives from someone who misses me.",
        "The crowd sings along with me.",
        "My coworkers save me a seat at lunch.",
        "Someone I admire says my work changed their life.",
    ]),
    "H4": ("Moral fulfilment & meaning", [
        "I return the lost wallet with every dollar inside.",
        "I stand up for the quiet student and the bullying stops.",
        "I keep my promise even though no one would know.",
        "I donate my savings to a family in need.",
        "I tell the truth and it earns their trust.",
        "My work helps a patient walk again.",
        "I forgive my brother and the weight lifts.",
        "I refuse the shortcut and keep my integrity.",
        "I spend the afternoon teaching a child to read.",
        "My choices finally match everything I believe in.",
        "I give credit to the colleague who earned it.",
        "I speak up for the person nobody listens to.",
        "I admit my mistake and make it right.",
        "My small kindness turns someone's whole day around.",
        "I plant the trees that my grandchildren will sit under.",
        "I stay loyal to my values when it is hard.",
        "I walk away from the deal that would have hurt others.",
        "I volunteer at the shelter and it feels like my purpose.",
        "I do the right thing and sleep soundly.",
        "I look back on this year and know I lived well.",
    ]),
    "H5": ("Cognitive pleasure & flow", [
        "The proof clicks into place and everything makes sense.",
        "I solve the puzzle after hours of effort.",
        "The code compiles and runs perfectly on the first try.",
        "I lose track of time while the music flows from my hands.",
        "A tricky idea suddenly becomes clear to me.",
        "I finally understand the whole argument.",
        "The pieces of the plan fit together beautifully.",
        "I master the hardest passage of the piece.",
        "My mind fills with exciting new connections.",
        "A new theory explains all the strange results.",
        "I follow the elegant logic to its end.",
        "The problem that stumped me for weeks gives way.",
        "I read a book that opens up a whole new world.",
        "The sentence I was hunting for arrives, exactly right.",
        "I fix the last bug and the system hums.",
        "I learn a new word and see it everywhere.",
        "Every question I ask leads somewhere fascinating.",
        "The chess position opens up and I see the winning idea.",
        "My understanding deepens with each page I turn.",
        "The design comes together exactly as I imagined.",
    ]),
}

# --- the 5 matched controls (each shares ONE property with wellbeing, lacks wellbeing) -----

CONTROLS = {
    "B": ("Negative emotion", [   # personal, emotional, first-person - but the wrong valence
        "The footsteps behind me get closer.",
        "My landlord's rudeness makes me furious.",
        "The smell of rotten food turns my stomach.",
        "I sit alone in the dark and cry quietly.",
        "My hands shake as I open the results letter.",
        "The driver cuts me off and I seethe.",
        "A hollow sadness settles over my evening.",
        "I dread tomorrow's meeting all night.",
        "The slimy mess in the sink disgusts me.",
        "My boss takes credit for my work and I fume.",
        "I stare at the empty chair where she used to sit.",
        "A cold panic rises as the plane shakes.",
        "The noisy neighbors keep me irritated until dawn.",
        "I feel numb and heavy as the rain falls.",
        "The spider crawls toward my pillow.",
        "Another delayed flight leaves me angry.",
        "Loneliness presses down on me in the quiet house.",
        "I flinch at every sound in the dark hallway.",
        "The rude waiter leaves me bitter and annoyed.",
        "I miss them terribly and the evening drags.",
    ]),
    "C1": ("Positive world state", [   # good valence, but impersonal: nobody's feelings are described
        "I read that the new bridge opens to the public today.",
        "The harvest this year is the largest on record.",
        "I hear the city plans to plant ten thousand new trees.",
        "Unemployment falls to its lowest level in decades.",
        "The river runs clean again after the cleanup.",
        "I read that the hospital opened a new wing for children.",
        "The village gets electricity for the first time.",
        "The endangered species recovers across the region.",
        "I see that crime in the district dropped sharply this year.",
        "The library receives a generous donation of books.",
        "I read that scientists announced a cure for the rare disease.",
        "The railway cuts travel time in half.",
        "The festival draws record crowds this summer.",
        "I read that the company reports strong profits this quarter.",
        "The school wins funding for new classrooms.",
        "The reservoir is full after heavy rain.",
        "I hear the park reopens with new playgrounds.",
        "The vaccine program reaches every remote town.",
        "Wildflowers cover the hills in spring.",
        "The old castle is restored to its former glory.",
    ]),
    "C2": ("Wanting without having", [   # approach / anticipation / striving, but the pleasure hasn't arrived
        "I stare at the house I cannot afford.",
        "I count the days until the trip.",
        "I scroll through the listings, wanting a bigger place.",
        "I rehearse my speech again before tomorrow.",
        "I save every spare coin for the new laptop.",
        "I press my nose to the bakery window, hungry.",
        "I crave something sweet and cannot find it.",
        "I wait impatiently for the results to arrive.",
        "I plan every detail of the upcoming wedding.",
        "I want the promotion more than anything.",
        "I envy my neighbor's new car.",
        "I pace the platform waiting for the late train.",
        "I long for the weekend to start.",
        "I refresh the page hoping for good news.",
        "I chase the deadline with three days to go.",
        "I circle the store, undecided about what to buy.",
        "I wish I had been invited to the party.",
        "I keep checking my phone for her reply.",
        "I train every morning for a race months away.",
        "I look at the menu, wanting everything on it.",
    ]),
    "D": ("Neutral", [
        "I fold the towels and put them in the closet.",
        "The train enters the station.",
        "I write my address at the top of the form.",
        "The clock on the wall reads half past four.",
        "I carry the boxes to the second floor.",
        "The printer produces ten pages.",
        "I park the car in the garage.",
        "The meeting is scheduled for Tuesday.",
        "I check the temperature on the thermostat.",
        "The bus stops at the corner.",
        "I put the receipts in the drawer.",
        "The door opens inward.",
        "I copy the file to a new folder.",
        "The window faces the parking lot.",
        "I count the chairs around the table.",
        "The package arrives on Thursday.",
        "I sort the mail into two piles.",
        "The shelf holds six books.",
        "I turn off the hallway light.",
        "The road continues straight for a mile.",
    ]),
    "E": ("Neutral body sensation", [   # bodily, first-person, but no pleasure
        "My left foot rests flat on the floor.",
        "I notice my heartbeat while sitting still.",
        "My hands are slightly dry from the air.",
        "I blink as the screen brightness changes.",
        "My stomach growls before lunch.",
        "I swallow and clear my throat.",
        "My shirt sleeve brushes against my wrist.",
        "I shift my weight from one leg to the other.",
        "My eyelids feel heavy as the evening goes on.",
        "I sneeze twice in a row.",
        "My breath fogs in the cold air.",
        "My fingers tap against the desk.",
        "I scratch an itch on my shoulder.",
        "My hair falls across my forehead.",
        "I feel the chair press against my back.",
        "My mouth feels dry in the heat.",
        "I stretch my arms over my head.",
        "My glasses slide down my nose.",
        "I rub my eyes after the long drive.",
        "The seat belt rests across my chest.",
    ]),
}

POSITIVE_CATS = list(WELLBEING)
CONTROL_CATS = list(CONTROLS)
LABELS = {**{k: v[0] for k, v in WELLBEING.items()}, **{k: v[0] for k, v in CONTROLS.items()}}
NEUTRAL_CAT = "D"

# --- standalone validation sets (never used to build the vector) --------------------------

# High arousal / intensity, but no wellbeing: tests "happiness" vs "activation".
AROUSAL = [
    "My heart pounds as I sprint for the closing doors.",
    "I race the clock to finish before the deadline.",
    "Adrenaline surges through me as the siren sounds.",
    "I slam the brakes at the last possible second.",
    "The crowd roars and my pulse hammers in my ears.",
    "I scramble up the rocks as the tide rushes in.",
    "My breath comes fast as I dash across the street.",
    "I grip the wheel hard as the car skids.",
    "I jump out of the way as the bike flies past.",
    "My whole body tenses as the countdown begins.",
    "I dive for the ball a split second too late.",
    "The alarm blares and I bolt out of bed.",
    "I fight to keep my balance on the swaying deck.",
    "My legs burn as I climb the last flight of stairs.",
    "I weave through the packed crowd at top speed.",
    "The starting gun fires and I launch forward.",
    "I catch the falling glass an inch from the floor.",
    "My thoughts race as the clock ticks down.",
    "I hammer the keys, racing to save the file.",
    "I push the heavy door shut as the wind howls.",
    "The engine screams as I take the sharp corner.",
    "I grab the rope and haul with everything I have.",
    "Sparks fly as I rush to fix the line.",
    "I tackle the problem at full speed with no time to think.",
    "My hands move fast as the orders pile up.",
    "I leap the fence as the dog charges.",
    "The buzzer sounds and everyone scrambles.",
    "I sprint the last hundred meters with nothing left.",
    "I shout over the noise to be heard.",
    "The lights flash and I scramble for the exit.",
]

# "Numb" analogue: pleasant situations where the pleasure is explicitly absent.
ANHEDONIC = [
    "I taste the warm cake but feel nothing at all.",
    "The sunny beach does nothing for me today.",
    "My friends throw me a party and I feel nothing.",
    "The hot bath leaves me completely indifferent.",
    "I win the prize and it means nothing to me.",
    "I hear the good news without feeling anything.",
    "The music I once loved leaves me cold.",
    "I hug my sister and feel no warmth.",
    "The delicious meal tastes like cardboard to me.",
    "The praise washes over me without effect.",
    "I finish the project and feel no satisfaction.",
    "The beautiful sunset leaves me unmoved.",
    "I open the gift and feel absolutely nothing.",
    "The massage goes on and I feel no relief.",
    "My favorite show plays and I feel empty.",
    "I stand in the spring garden, feeling flat and detached.",
    "They sing to me and I feel no connection.",
    "I reach the summit and feel nothing at all.",
    "The compliments arrive but none of them land.",
    "I sip my favorite tea and it tastes of nothing.",
    "The puppy jumps into my lap and I feel nothing.",
    "I get the promotion and shrug it off.",
    "The vacation starts and I feel no pleasure.",
    "The soft bed means nothing to me tonight.",
    "I finally solve the riddle and feel completely empty.",
    "The festival glows around me, and I feel nothing.",
    "A warm breeze touches my face and I feel numb.",
    "I laugh on cue, but nothing inside me moves.",
    "Everyone cheers for me while I feel hollow.",
    "I eat my favorite dish and feel no joy in it.",
]


# --- first-person -> third-person ("they") -------------------------------------------------

_RULES = [  # applied in order; word-boundary regexes, sentence-initial capitalisation handled after
    (r"\bI'm\b", "they're"), (r"\bI've\b", "they've"), (r"\bI'll\b", "they'll"), (r"\bI'd\b", "they'd"),
    (r"\bI am\b", "they are"), (r"\bI was\b", "they were"), (r"\bI\b", "they"),
    (r"\bmyself\b", "themselves"), (r"\bmine\b", "theirs"), (r"\b[Mm]y\b", "their"), (r"\b[Mm]e\b", "them"),
]


def to_third_person(sentence: str) -> str:
    """Rewrite a first-person sentence with the plural-form "they" (so verbs need no change)."""
    s = sentence
    for pat, rep in _RULES:
        s = re.sub(pat, rep, s)
    return s[0].upper() + s[1:]


def build_s2():
    """Return the dataset as JSON-able dict: {"1P": [...], "3P": [...]} of rows."""
    out = {"1P": [], "3P": []}
    for group in (WELLBEING, CONTROLS):
        for cat, (_, sents) in group.items():
            assert len(sents) == 20, (cat, len(sents))
            for i, s in enumerate(sents, start=1):
                out["1P"].append({"category": cat, "set": i, "sentence": s, "prompt": s + SUFFIX})
                t = to_third_person(s)
                out["3P"].append({"category": cat, "set": i, "sentence": t, "prompt": t + SUFFIX})
    return out
