import matplotlib.pyplot as plt
import sys

y = []
# 1000 points/frame
y1000 = []
# 500 points/frame
y500 = [-1.54741e-06,-0.00146421,0.000208658,-0.000884259,0.000398702,-0.000789207,-0.000655869,0.000640057,0.000300216,-6.35966e-05,-3.46132e-05,-0.000441692,-0.000838064,0.000761744,-0.00139397,0.00068746,-0.000150066,0.000557628,0.00222003,0.00245192,-6.11404e-05,-0.000871322,0.000822835,0.00148485,0.000709458,-0.000315565,-0.000734733,0.00124159,0.000468098,0.00127703,0.000348316,0.000688525,-0.000957205,0.00190097,0.00141658,0.00285757,0.00155605,0.000654905,0.0015627,0.00230473,0.00195687,-0.000221519,0.00237206,0.00311067,0.000514103,0.00233417,0.00375988,0.00318277,0.00140187,0.00128003,0.00225086,0.000123994,0.00215891,0.000985064,0.00253076,0.00134679,0.00132511,0.000688824,-0.000538933,0.00120723,0.000103247,-0.00109921,-0.0071075,-0.0119703,-0.0179475,-0.0282752,-0.0389016,-0.0503755,-0.0645524,-0.0757668,-0.0895623,-0.100695,-0.113614,-0.132444,-0.143917,-0.162539,-0.178321,-0.194354,-0.210799,-0.228764,-0.243794,-0.261465,-0.27779,-0.295329,-0.309307,-0.325023,-0.337704,-0.351164,-0.364732,-0.377807,-0.391682,-0.403577,-0.414602,-0.423126,-0.425545,-0.422724,-0.422258,-0.42274,-0.422593,-0.424326,-0.424922,-0.422728,-0.423746,-0.419435,-0.417099,-0.411631,-0.405312,-0.398719,-0.389784,-0.380905,-0.371101,-0.360382,-0.34976,-0.33832,-0.326831,-0.313903,-0.301564,-0.287556,-0.271579,-0.257199,-0.240021,-0.226908,-0.213391,-0.200201,-0.18663,-0.176975,-0.165695,-0.153972,-0.140456,-0.132104,-0.127948,-0.113879,-0.106369,-0.0975684,-0.0890841,-0.0815793,-0.0739303,-0.0660575,-0.060313,-0.05482,-0.0493463,-0.042915,-0.0379931,-0.0340245,-0.0300309,-0.0260883,-0.0211893,-0.0183766,-0.0149422,-0.011226,-0.0100259,-0.0065855,-0.00474509,-0.00424749,-0.00171771,-0.000500612,0.000233354,0.000106296,0.00174127,-0.000504455,-0.00113254,-0.00141106,-0.000339625,-0.00258325,-0.00341376,-0.00198762,-0.00460664,-0.00943109,-0.0114621,-0.0143694,-0.0188523,-0.0236922,-0.0287809,-0.0338399,-0.0389437,-0.0444458,-0.0490206,-0.0548763,-0.0616056,-0.0656434,-0.0702353,-0.07475,-0.0808058,-0.0879606,-0.0916882,-0.0965528,-0.103146,-0.108463,-0.113951,-0.12132,-0.122759,-0.135793,-0.140141,-0.14838,-0.152714,-0.16109,-0.167288,-0.172209,-0.179811,-0.187144,-0.197272,-0.206735,-0.209102,-0.21601,-0.22324,-0.225841,-0.233718,-0.237559,-0.2515,-0.253383,-0.266934,-0.274062,-0.283325,-0.290488,-0.299655,-0.308955,-0.316378,-0.325,-0.332159,-0.340181,-0.347317,-0.354765,-0.360757,-0.367158,-0.372207,-0.376502,-0.380092,-0.382966,-0.383814,-0.382327,-0.382184,-0.379187,-0.376286,-0.372338,-0.367462,-0.362289,-0.356226,-0.350565,-0.344029,-0.337271,-0.330103,-0.323899,-0.317065,-0.310356,-0.302331,-0.293826,-0.287975,-0.281195,-0.272924,-0.262687,-0.257135,-0.250056,-0.242173,-0.233095,-0.228772,-0.216407,-0.216537,-0.204562,-0.197813,-0.196562,-0.186691,-0.183025,-0.176729,-0.172104,-0.169231,-0.163288,-0.159209,-0.150601,-0.147381,-0.137337,-0.134757,-0.132322,-0.124493,-0.117605,-0.114445,-0.107821,-0.103861,-0.0974596,-0.0907347,-0.0883757,-0.0820813,-0.0748629,-0.0726443,-0.069725,-0.0626123,-0.0580184,-0.0539079,-0.0498107,-0.0467441,-0.0414316,-0.0369048]

# 100 points/frame
y100 = []

# 33 points/frame
y33 = [-1.46374e-05,0.000513449,0.000914618,-0.000257314,0.00085326,0.000340567,0.00109021,0.00131469,0.000728816,0.00141673,0.00161394,0.00149528,0.000399313,0.00149055,0.00194646,0.00246513,0.00143117,0.00242232,0.00165008,0.00247355,0.00256771,0.00166587,0.000605754,0.00137605,0.00149917,0.000874419,0.00126981,0.0016024,0.000118522,0.00105014,0.00068307,0.00206395,0.000933223,0.000660353,0.00115943,0.00283561,0.00142851,0.00279244,0.00246068,0.00249958,0.0028211,0.00268443,0.00307286,0.00242002,0.0033094,0.00174313,0.00188739,0.00135404,0.00217141,0.002433,0.00223434,0.00248829,0.00106899,0.00252426,0.00297869,0.00158297,0.00140539,0.00117133,0.00134283,0.000234631,-9.92219e-05,-0.00267254,-0.00718713,-0.0112993,-0.0185157,-0.0279977,-0.037482,-0.050769,-0.0626724,-0.0735938,-0.0868832,-0.0916295,-0.110505,-0.129618,-0.141548,-0.158654,-0.167403,-0.202152,-0.216384,-0.229852,-0.255393,-0.271749,-0.285257,-0.295544,-0.312286,-0.32703,-0.3393,-0.353303,-0.366218,-0.380566,-0.384773,-0.397904,-0.408574,-0.421601,-0.427472,-0.429085,-0.425675,-0.425812,-0.424019,-0.423009,-0.42197,-0.421499,-0.420497,-0.414861,-0.41226,-0.406199,-0.401698,-0.393555,-0.384436,-0.382823,-0.372651,-0.362014,-0.352721,-0.341278,-0.329487,-0.315473,-0.30662,-0.291325,-0.279185,-0.259268,-0.247164,-0.234937,-0.21645,-0.202979,-0.195078,-0.16898,-0.164863,-0.148055,-0.1434,-0.130223,-0.119552,-0.106871,-0.105145,-0.0986858,-0.0823585,-0.0823676,-0.0757415,-0.0648071,-0.0625204,-0.0542839,-0.0504305,-0.0452876,-0.0392008,-0.0361852,-0.0296156,-0.0258543,-0.0216469,-0.0178844,-0.0135198,-0.0109459,-0.0103992,-0.00644584,-0.00475241,-0.00175517,-0.00167818,0.000105959,-0.00289324,0.00101838,0.00117835,0.00105098,0.000431134,0.00122287,0.00121943,0.000168603,-0.00166669,-0.00205068,-0.0054746,-0.00710969,-0.00950434,-0.0146958,-0.0186551,-0.0253929,-0.0277651,-0.035139,-0.0406088,-0.0471264,-0.0498416,-0.0543098,-0.0619377,-0.0678235,-0.063696,-0.0748173,-0.0816722,-0.0825541,-0.0827192,-0.0987152,-0.0929329,-0.107373,-0.11235,-0.12031,-0.120806,-0.134016,-0.135151,-0.145171,-0.147182,-0.158173,-0.168035,-0.1701,-0.190808,-0.186807,-0.199742,-0.198641,-0.20961,-0.226456,-0.231256,-0.23632,-0.24118,-0.241983,-0.257998,-0.253631,-0.265784,-0.280509,-0.287943,-0.294452,-0.30088,-0.313355,-0.319243,-0.325858,-0.333007,-0.342803,-0.350211,-0.355794,-0.362671,-0.369066,-0.374616,-0.378366,-0.381234,-0.384604,-0.386543,-0.385046,-0.384709,-0.38001,-0.377657,-0.374721,-0.369531,-0.364014,-0.3581,-0.352501,-0.345462,-0.34015,-0.331095,-0.32668,-0.317667,-0.313955,-0.307158,-0.2986,-0.291824,-0.284779,-0.278002,-0.272141,-0.263583,-0.245624,-0.252169,-0.234037,-0.233558,-0.227861,-0.223689,-0.20719,-0.20776,-0.19638,-0.191666,-0.187109,-0.183166,-0.173042,-0.16915,-0.160273,-0.158977,-0.146546,-0.141708,-0.140595,-0.132544,-0.130612,-0.12157,-0.120563,-0.112998,-0.10792,-0.0968289,-0.0972579,-0.0907633,-0.0805008,-0.0816449,-0.0715228,-0.0760443,-0.0669965,-0.0608387,-0.0595424,-0.053723,-0.0504026,-0.0453797,-0.0424901,-0.0382106,-0.0349365,-0.0299216,-0.0249591,-0.0219145,-0.0180405,-0.0132361,-0.0101393,-0.00660476,-0.00523973]
x = [i for i in range(len(y500))]

fig2 = plt.figure()

ax = fig2.add_subplot()
twin = ax.twinx()
twin2 = ax.twinx()
twin3 = ax.twinx()
a33, = twin2.plot(x, y33[:len(y500)], color='red', label='33')
a500, = ax.plot(x, y500, color='green', label='500')
# a1000, = twin.plot(x, y1000, color='blue', label='1000')
# a100, = twin3.plot(x, y100, color='orange', label='100')
ax.set_xlabel('iteration')
fig2.legend()
plt.show()
sys.exit()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

file = open("/home/arnav/biswas/students/cs378_starter_blank/rev3.csv", "r")


def in_range(data, i):
    return (data[i][0] > 0.4 and data[i][0] < 0.9) and (
        data[i][1] > -0.19 and data[i][1] < 0.05) and (data[i][2] > 0
                                                      and data[i][2] < 0.6)


data = [[
    float(i.split(",")[0]),
    float(i.split(",")[1]),
    float(i.split(",")[2])
] for i in file.readlines()]

xs = [
    data[i][0] for i in range(len(data)) if i % 500 == 0 and in_range(data, i)
]

ys = [
    data[i][1] for i in range(len(data)) if i % 500 == 0 and in_range(data, i)
]

zs = [
    data[i][2] for i in range(len(data)) if i % 500 == 0 and in_range(data, i)
]

xsa = [
    data[i][0] for i in range(len(data))
    if i % 500 == 0 and not in_range(data, i)
]

ysa = [
    data[i][1] for i in range(len(data))
    if i % 500 == 0 and not in_range(data, i)
]

zsa = [
    data[i][2] for i in range(len(data))
    if i % 500 == 0 and not in_range(data, i)
]

print(len(xs))

ax.scatter(xs, ys, zs)
ax.scatter(xsa, ysa, zsa)
ax.scatter([0], [0], [0])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()