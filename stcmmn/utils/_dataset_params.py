DATASET_PARAMS = {
    'BNCI2014001': {
        'subject_id': None,
        'fs': 250,
        'channels': [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2',
            'CP4', 'P1', 'Pz', 'P2', 'POz',
        ],
        'mapping': {
            "left_hand": 0,
            "right_hand": 1,
            "feet": 3,
            "tongue": 2,
        }
    },
    'BNCI2014004': {
        'subject_id': None,
        'fs': 250,
        'channels': ['C3', 'Cz', 'C4'],
        'mapping': {
            "left_hand": 0,
            "right_hand": 1,
        }
    },
    'BNCI2015001': {
        'subject_id': None,
        'fs': 512,
        'channels': [
            'FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4',
        ],
        'mapping': {
            "right_hand": 0,
            "feet": 1,
        }
    },
    'PhysionetMI': {
        'subject_id': [i for i in range(1, 109) if i not in [
            34, 37, 41, 51, 64, 72,
            73, 74, 76, 88, 89, 92,
            100, 102, 104
        ]],
        'fs': 160,
        'channels': [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2',
            'CP4', 'P1', 'Pz', 'P2', 'POz',
        ],
        'mapping': {
            "left_hand": 1,
            "right_hand": 0,
            "feet": 2,
            "hands": 3,
        }
    },
    'Cho2017': {
        'subject_id': None,
        'fs': 512,
        'channels': [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2',
            'CP4', 'P1', 'Pz', 'P2', 'POz',
        ],
        'mapping': {
            "left_hand": 0,
            "right_hand": 1,
        }
    },
    'Weibo2014': {
        'subject_id': None,
        'fs': 200,
        'channels': [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2',
            'CP4', 'P1', 'Pz', 'P2', 'POz',
        ],
        'mapping': {
            "left_hand": 0,
            "right_hand": 1,
        }
    },
}
