"""Small, fixed probes; illustrative rather than a comprehensive benchmark."""

PROMPTS = [
    'The capital of France is',
    'Artificial intelligence is',
    'Once upon a time, there was',
    'The purpose of a computer is',
    'In the future, humans will',
]

FACTS = [
    ('The capital of France is', [' Paris', ' London', ' Berlin', ' Madrid'], 0),
    ('The capital of Japan is', [' Seoul', ' Beijing', ' Tokyo', ' Bangkok'], 2),
    ('The capital of Italy is', [' Rome', ' Paris', ' Vienna', ' Lisbon'], 0),
    ('The capital of Germany is', [' Berlin', ' Rome', ' Madrid', ' Moscow'], 0),
    ('The capital of Canada is', [' Toronto', ' Ottawa', ' Vancouver', ' Montreal'], 1),
    ('The capital of Spain is', [' Madrid', ' Lisbon', ' Berlin', ' Paris'], 0),
    ('Water freezes at', [' 0 degrees Celsius', ' 100 degrees Celsius', ' 50 degrees Celsius'], 0),
    ('The largest ocean on Earth is the', [' Atlantic Ocean', ' Indian Ocean', ' Pacific Ocean'], 2),
    ('The chemical symbol for gold is', [' Au', ' Ag', ' Fe', ' Cu'], 0),
    ('The chemical symbol for oxygen is', [' O', ' H', ' N', ' C'], 0),
]

GRAMMAR = [
    ('The children are playing outside.', 'The children is playing outside.'),
    ('She goes to school every day.', 'She go to school every day.'),
    ('They were happy yesterday.', 'They was happy yesterday.'),
    ('The dog runs quickly.', 'The dog run quickly.'),
    ('He has two brothers.', 'He have two brothers.'),
    ('I have seen this movie.', 'I has seen this movie.'),
    ('There are five books on the desk.', 'There is five books on the desk.'),
    ('My friends enjoy reading.', 'My friends enjoys reading.'),
    ('She does not like tea.', 'She do not like tea.'),
    ('We are going home.', 'We is going home.'),
]
