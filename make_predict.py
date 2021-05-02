import extract_features
import numpy as np
def get_pred(model, input_csv):
    classifier, normalizer = model
    print("START")
    print(input_csv)
    print("DONE")
    input_fv = extract_features.get_fv_csv(input_csv)
    pred_arr = []
    print(input_fv.shape)
    print("^^ input_fv")
    for fv in input_fv:
        print(fv.shape)
        print("fv.shape in input_fv")
        norm_inp = normalizer.transform(np.array([fv,]))
        print(norm_inp)
        pred = classifier.predict(norm_inp)
        pred_arr.append(pred)
    return str(np.array(pred_arr).reshape(-1).tolist())

if __name__ == "__main__":
    import pickle
    with open('best_classifier.pkl', 'rb') as fil:
        classifier = pickle.load(fil)
    with open('normalizer.pkl', 'rb') as fil:
        normalizer = pickle.load(fil)

    input_csv = '''-0.8719109,5.183293,8.493409,8.4963989E14
-0.7229122,5.2618647,7.8336935,8.4964009E14
-0.21650806,5.2048526,8.099592,8.496403E14
-0.043554474,5.1588593,9.181869,8.496405E14
0.32774463,6.046145,7.2357826,8.496407E14
0.6626326,7.4206705,3.4782348,8.496409E14
0.31385094,5.252762,8.408609,8.496411E14
0.035496294,3.756546,11.04124,8.496413E14
0.08867593,5.4463167,7.3660965,8.496415E14
0.6794009,7.7114806,2.8961334,8.496417E14
0.008187843,5.841571,7.027855,8.4964191E14
-0.30274525,3.809726,10.673773,8.4964211E14
-0.056010973,4.696532,9.396983,8.4964231E14
0.41877285,7.295148,4.628544,8.4964251E14
0.5917264,7.677945,3.013033,8.4964271E14
-0.07182113,5.442484,8.779908,8.4964291E14
-0.1954278,4.180546,9.92782,8.4964311E14
0.02160256,5.5454893,8.611746,8.4964332E14
0.21898994,7.1653123,4.215084,8.4964352E14
-0.16476569,6.6244125,5.5431376,8.4964372E14
-0.26058483,4.887212,10.045679,8.4964392E14
-0.09338044,4.536035,9.617368,8.4964412E14
-0.4982163,6.2655706,6.38922,8.4964432E14
-0.22369447,6.701547,4.711427,8.4964452E14
0.057055615,4.868048,9.58431,8.4964472E14
0.3282238,4.402846,8.709002,8.4964493E14
-0.67164886,2.7782328,9.766366,8.4965217E14
-0.59259814,4.1197004,8.9111805,8.4965238E14
-0.12116798,5.373015,6.242139,8.4965258E14
0.13562731,3.1864219,9.4525585,8.4965278E14
0.41973108,2.2311053,10.694853,8.4965298E14
0.14425103,3.4614234,8.723854,8.4965318E14
0.26258764,5.0213585,7.101636,8.4965338E14
0.32247466,3.1897762,9.060179,8.4965358E14
0.21707353,1.9632909,10.519983,8.4965378E14
0.5466915,2.9176495,9.949381,8.4965399E14
0.44991413,5.212039,7.276506,8.4965419E14
0.45183048,6.314439,5.195793,8.4965439E14
0.7613264,3.911773,8.649594,8.4965459E14
0.5481287,1.4501793,11.287494,8.4965479E14
0.6161603,2.593302,10.0749035,8.4965499E14
0.2549221,4.386557,8.906869,8.4965519E14
-0.17770128,6.4279838,5.8880863,8.4965539E14
0.06999119,6.142443,3.7838979,8.496556E14
0.18497415,4.5949636,9.023767,8.496558E14
-0.186325,3.439385,10.703478,8.49656E14
-0.61655295,3.0642529,9.037662,8.496562E14
-0.6448196,3.1097667,9.230257,8.496564E14
;
0.19618991,0.058433034,-0.10543446,8.4963996E14
0.015947327,-0.33582094,-0.094675295,8.4964016E14
-0.23652013,-0.29342347,-0.019254642,8.4964036E14
0.6408309,-0.2650875,-0.0019973698,8.4964056E14
1.0823826,0.050869662,0.05904223,8.4964077E14
-0.72473043,0.13523851,-0.09851025,8.4964097E14
-1.5816282,-0.18380792,-0.13504878,8.4964117E14
0.9726604,0.055130728,-0.0033822102,8.4964137E14
1.5948808,0.1734815,0.09600687,8.4964157E14
-0.56877583,-0.023911834,-0.07816371,8.4964177E14
-1.8070382,0.013905017,-0.18106818,8.4964197E14
0.07144754,-0.0149636045,0.027297374,8.4964217E14
1.9995744,-0.25049338,0.17078836,8.4964238E14
0.8870133,0.105198085,0.0666056,8.4964258E14
-1.4006402,0.44949126,-0.100747295,8.4964278E14
-1.4821327,-0.40793934,-0.21952419,8.4964298E14
0.5509227,0.10338715,0.0055660014,8.4964318E14
1.407714,0.08336021,0.022610217,8.4964338E14
0.19171575,0.16059181,-0.01925464,8.4964358E14
-1.5066339,-0.19062561,-0.16061509,8.4964379E14
-0.45170337,-0.0039914064,-0.016165372,8.4964399E14
1.060012,0.35510892,0.06894918,8.4964419E14
0.5984334,0.0059155445,0.054674648,8.4964439E14
-1.0531511,-0.32346395,-0.04237085,8.4964459E14
-0.5845418,-0.27020076,0.043489385,8.4964479E14
0.58458495,0.05395893,0.06916223,8.4965217E14
1.0246454,-0.13065125,0.09121318,8.4965238E14
-0.75562304,-0.2555001,-0.09361003,8.4965258E14
-1.168093,-0.25326306,-0.054195274,8.4965278E14
0.63635683,0.17167054,-0.046844963,8.4965298E14
0.89042205,0.009963545,0.056485604,8.4965318E14
-0.3600907,0.043306287,-0.006791055,8.4965338E14
-0.9965856,-0.11211565,0.009827052,8.4965358E14
0.121301875,-0.08622975,-0.03896201,8.4965378E14
1.4214561,-0.011554781,-0.02884201,8.4965399E14
0.8577186,0.11052441,0.0075900014,8.4965419E14
-0.7990858,-0.11222218,0.015472953,8.4965439E14
-1.9194233,0.0019740588,-0.010626003,8.4965459E14
0.35331628,-0.04745417,-0.008815056,8.4965486E14
0.8913808,0.11755516,-0.066871926,8.4965506E14
1.3649971,0.32453588,-0.030439908,8.4965526E14
0.6593665,0.07547726,0.013768531,8.4965546E14
-0.852349,-0.38215995,-0.07219824,8.4965566E14
-1.3832765,-0.03786679,-0.008282425,8.4965586E14
-0.2026447,0.5142593,0.05446159,8.4965607E14
-0.040192056,-0.02178131,0.0022636848,8.4965627E14
;
-14.30949,-39.63336,-34.83293,8.4964003E14
-14.780034,-39.619324,-33.0688,8.4964023E14
-15.109193,-39.192444,-34.040573,8.4964043E14
-15.010431,-43.82019,-30.211693,8.4964063E14
-14.894935,-48.15631,-20.308334,8.4964083E14
-14.271762,-47.893707,-22.402603,8.4964103E14
-16.923164,-36.489655,-34.08316,8.4964124E14
-16.291632,-38.664795,-31.670265,8.4964144E14
-12.934818,-41.82068,-12.817848,8.4964164E14
-11.981041,-41.84253,-12.824165,8.4964184E14
-13.283568,-34.116516,-26.188812,8.4964204E14
-13.2026415,-30.615875,-29.983154,8.4964224E14
-13.008543,-38.73938,-17.55832,8.4964244E14
-12.560051,-41.155334,-5.403412,8.4964264E14
-12.177335,-37.664734,-12.770866,8.4964285E14
-10.688563,-27.213135,-25.508179,8.4964305E14
-10.027957,-28.632599,-25.78807,8.4964318E14
-8.455811,-36.300293,-14.992912,8.4964338E14
-8.731964,-37.54785,-9.827919,8.4964358E14
-8.122774,-33.646576,-21.025452,8.4964379E14
-10.962776,-27.13379,-26.335075,8.4964399E14
-7.5102487,-32.680664,-22.501602,8.4964419E14
-5.649629,-37.34198,-15.002258,8.4964439E14
-6.3571897,-34.38681,-19.894165,8.4964459E14
-10.777906,-28.541779,-26.291634,8.4964479E14
-10.330341,-19.012878,-29.960278,8.4965217E14
-9.54306,-26.88858,-27.954346,8.4965238E14
-8.6486225,-31.675812,-24.606537,8.4965258E14
-12.166212,-22.75998,-29.347244,8.4965278E14
-12.27446,-17.144928,-30.805931,8.4965298E14
-11.634834,-12.898376,-33.982555,8.4965318E14
-10.890434,-18.102692,-30.300186,8.4965338E14
-13.264479,-10.756012,-34.35895,8.4965358E14
-14.321192,-4.361664,-35.574196,8.4965378E14
-13.584581,-9.711578,-34.76897,8.4965399E14
-11.598158,-20.971222,-25.93988,8.4965419E14
-10.898367,-23.514801,-22.443878,8.4965439E14
-13.283199,-16.100983,-32.028618,8.4965459E14
-19.987429,-1.9127502,-28.30796,8.4965479E14
-13.713466,-11.169617,-36.06124,8.4965499E14
-10.184073,-20.788177,-31.618477,8.4965519E14
-7.7990513,-27.562653,-21.654205,8.4965539E14
-7.6058226,-29.108215,-20.005432,8.496556E14
-10.127351,-23.452728,-29.452774,8.496558E14
-12.055238,-13.229645,-36.372974,8.49656E14
-11.1345625,-11.425629,-36.277046,8.496562E14
-11.620946,-10.710815,-35.3356,8.496564E14
;'''
    print("________________________________"*2)

    print(get_pred((classifier, normalizer), input_csv))
