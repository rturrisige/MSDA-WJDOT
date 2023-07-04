##
# COMMAND MAPPING

dict_CL = {'No.': [0], 'Si.': [1], 'Chiama.': [2], 'Ciao.': [3], 'Ci vediamo domani.': [4], 'Come stai?': [5],
           'Devo andare in bagno.': [6], 'Grazie.': [7], 'Ho bisogno di lavarmi.': [8],
           'Ho caldo.': [9], 'Ho dolore.': [10], 'Ho fame.': [11], 'Ho freddo.': [12], 'Ho sete.': [13],
           'Ho sonno.': [14], 'Ho troppa saliva.': [15], 'Passami il telefono.': [16], 'Prego.': [17],
           'Salutami ...': [18], 'Sono arrabbiato': [19], 'Sono triste.': [20], 'Ti voglio bene.': [21],
           'Tutto bene?': [22], 'Voglio cambiare posizione.': [23], 'Voglio stare solo.': [24]}


def map_label2command(labels):
    commands = [dict_CL.keys()[dict_CL.values().index(label)] for label in labels]
    return commands