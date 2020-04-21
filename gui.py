
import ipywidgets as widgets
import matplotlib.pyplot as plt
import ExoCMD as ExoCMD
from IPython.display import clear_output
from IPython.display import Javascript

style = {'description_width': 'initial'}

planets_path = widgets.Text(
    value='planet_fluxes.txt',
    placeholder='Paste here',
    description='Path to planet database:',
    disabled=False,
    style = style
)
BD_path = widgets.Text(
    value='bd_mags.txt',
    placeholder='Paste here',
    description='Path to brown dwarf database:',
    disabled=False,
    style = style
)


hl1 = widgets.Text(
    value = 'WASP-12',
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c1 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#E77923',
    disabled=False
)


hl2 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)

c2 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Colour',
    style = style,
    value = '#D8E50E'
)

c2 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#D8E50E',
    disabled=False
)


hl3 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c3 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#5FD61C',
    disabled=False
)


hl4 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c4 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#1CD69E',
    disabled=False
)


hl5 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c5 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#1CA3D6',
    disabled=False
)


hl6 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c6 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#224AC4',
    disabled=False
)


hl7 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c7 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#7D22C4',
    disabled=False
)


hl8 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c8 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#C422BD',
    disabled=False
)


hl9 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c9 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#C42273',
    disabled=False
)


hl10 = widgets.Text(
    placeholder = 'Insert Here',
    description = 'Name of highlighted planet',
    style = style
)


c10 = widgets.ColorPicker(
    concise=False,
    description='Pick a color',
    value='#C4222C',
    disabled=False
)


band1 = widgets.Dropdown(
    options=['J', 'H', 'K', '3.6', '4.5', '5.8', '8.0'],
    value='J',
    description=' Colour Band 1:',
    disabled=False,
    style = style
)

band2 = widgets.Dropdown(
    options=['J', 'H', 'K', '3.6', '4.5', '5.8', '8.0'],
    value='H',
    description=' Colour Band 2:',
    disabled=False,
    style = style
)

band1_synth = widgets.Dropdown(
    options=['z', 'J', 'W', 'H', 'K', 'NB1190', 'NB2090'],
    value='J',
    description=' Colour Band 1:',
    disabled=False,
    style = style
)

band2_synth = widgets.Dropdown(
    options=['z', 'J', 'W', 'H', 'K', 'NB1190', 'NB2090'],
    value='H',
    description=' Colour Band 2:',
    disabled=False,
    style = style
)

absmag = widgets.Dropdown(
    options=['J', 'H', 'K', '3.6', '4.5', '5.8', '8.0'],
    value='K',
    description='Absolute Mag Band:',
    disabled=False,
    style = style
)

absmag_synth = widgets.Dropdown(
    options=['z', 'J', 'W', 'H', 'K', 'NB1190', 'NB2090'],
    value='K',
    description='Absolute Mag Band:',
    disabled=False,
    style = style
)

colour_by = widgets.Dropdown(
    options=['C/O Ratio', 'Metallicity', 'Spectral Type', 'Effective Temperature', 'Surface Gravity'],
    value='C/O Ratio',
    description='Colour points by:',
    disabled=False,
    style = style
)

CO_035 = widgets.Checkbox(
    value=False,
    description='0.35',
    disabled=False,
    indent=False
)

CO_055 = widgets.Checkbox(
    value=False,
    description='0.55',
    disabled=False,
    indent=False
)

CO_070 = widgets.Checkbox(
    value=False,
    description='0.70',
    disabled=False,
    indent=False
)

CO_071 = widgets.Checkbox(
    value=False,
    description='0.71',
    disabled=False,
    indent=False
)

CO_072 = widgets.Checkbox(
    value=False,
    description='0.72',
    disabled=False,
    indent=False
)

CO_073 = widgets.Checkbox(
    value=False,
    description='0.73',
    disabled=False,
    indent=False
)

CO_074 = widgets.Checkbox(
    value=False,
    description='0.74',
    disabled=False,
    indent=False
)

CO_075 = widgets.Checkbox(
    value=False,
    description='0.75',
    disabled=False,
    indent=False
)

CO_085 = widgets.Checkbox(
    value=False,
    description='0.85',
    disabled=False,
    indent=False
)

CO_090 = widgets.Checkbox(
    value=False,
    description='0.90',
    disabled=False,
    indent=False
)

CO_091 = widgets.Checkbox(
    value=False,
    description='0.91',
    disabled=False,
    indent=False
)

CO_092 = widgets.Checkbox(
    value=False,
    description='0.92',
    disabled=False,
    indent=False
)

CO_093 = widgets.Checkbox(
    value=False,
    description='0.93',
    disabled=False,
    indent=False
)

CO_094 = widgets.Checkbox(
    value=False,
    description='0.94',
    disabled=False,
    indent=False
)

CO_095 = widgets.Checkbox(
    value=False,
    description='0.95',
    disabled=False,
    indent=False
)

CO_100 = widgets.Checkbox(
    value=True,
    description='1.0',
    disabled=False,
    indent=False
)

CO_105 = widgets.Checkbox(
    value=False,
    description='1.05',
    disabled=False,
    indent=False
)

CO_112 = widgets.Checkbox(
    value=False,
    description='1.12',
    disabled=False,
    indent=False
)

CO_140 = widgets.Checkbox(
    value=False,
    description='1.40',
    disabled=False,
    indent=False
)

CO_all = widgets.Checkbox(
    value=False,
    description='All',
    disabled=False,
    indent=False
)


met__05 = widgets.Checkbox(
    value=False,
    description='-0.5',
    disabled=False,
    indent=False
)

met_00 = widgets.Checkbox(
    value=True,
    description='0.0',
    disabled=False,
    indent=False
)

met_05 = widgets.Checkbox(
    value=False,
    description='0.5',
    disabled=False,
    indent=False
)

met_10 = widgets.Checkbox(
    value=False,
    description='1.0',
    disabled=False,
    indent=False
)

met_20 = widgets.Checkbox(
    value=False,
    description='2.0',
    disabled=False,
    indent=False
)

met_all = widgets.Checkbox(
    value=False,
    description='All',
    disabled=False,
    indent=False
)

Teff_all = widgets.Checkbox(
    value=True,
    description='All',
    disabled=False,
    indent=False
)

Teff_10 = widgets.Checkbox(
    value=False,
    description='1000K',
    disabled=False,
    indent=False
)

Teff_12 = widgets.Checkbox(
    value=False,
    description='1250K',
    disabled=False,
    indent=False
)

Teff_15 = widgets.Checkbox(
    value=False,
    description='1500K',
    disabled=False,
    indent=False
)

Teff_17 = widgets.Checkbox(
    value=False,
    description='1750K',
    disabled=False,
    indent=False
)

Teff_20 = widgets.Checkbox(
    value=False,
    description='2000K',
    disabled=False,
    indent=False
)

Teff_22 = widgets.Checkbox(
    value=False,
    description='2250K',
    disabled=False,
    indent=False
)

Teff_25 = widgets.Checkbox(
    value=False,
    description='2500K',
    disabled=False,
    indent=False
)


spt_all = widgets.Checkbox(
    value=False,
    description='All',
    disabled=False,
    indent=False
)

spt_f = widgets.Checkbox(
    value=False,
    description='F5',
    disabled=False,
    indent=False
)

spt_g = widgets.Checkbox(
    value=True,
    description='G5',
    disabled=False,
    indent=False
)

spt_k = widgets.Checkbox(
    value=False,
    description='K5',
    disabled=False,
    indent=False
)

spt_m = widgets.Checkbox(
    value=False,
    description='M5',
    disabled=False,
    indent=False
)

logg_all = widgets.Checkbox(
    value=False,
    description='All',
    disabled=False,
    indent=False
)

logg_2 = widgets.Checkbox(
    value=True,
    description='2.3',
    disabled=False,
    indent=False
)

logg_3 = widgets.Checkbox(
    value=False,
    description='3.0',
    disabled=False,
    indent=False
)

logg_4 = widgets.Checkbox(
    value=False,
    description='4.0',
    disabled=False,
    indent=False
)

logg_5 = widgets.Checkbox(
    value=False,
    description='5.0',
    disabled=False,
    indent=False
)


bb09 = widgets.Checkbox(
    value=False,
    description='Blackbody Line',
    disabled=False,
    indent=False
)

adjusted = widgets.Checkbox(
    value=False,
    description='Adjusted Magnitudes',
    disabled=False,
    indent=False
)

colourbar = widgets.Checkbox(
    value=True,
    description='Add Colourbar',
    disabled=False,
    indent=False
)

poly = widgets.Checkbox(
    value=False,
    description='Show Polynomial',
    disabled=False,
    indent=False
)

bbmin = widgets.FloatSlider(
    value=1000,
    min=300,
    max=5000,
    step=50,
    description='BB min T (K)',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

bbmax = widgets.FloatSlider(
    value=5000,
    min=300,
    max=5000,
    step=50,
    description='BB max T (K)',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)


bbinc = widgets.FloatSlider(
    value=1000,
    min=100,
    max=1000,
    step=100,
    description='BB increment (K)',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)


bands = widgets.VBox([band1, band2, absmag])
bands_synth = widgets.VBox([band1_synth, band2_synth, absmag_synth])
properties = widgets.Box([bb09, adjusted, colourbar])
properties1 = widgets.Box([bb09, adjusted, colourbar, poly])
properties2 = widgets.Box([bb09, adjusted, colour_by])
bboptions = widgets.Box([bbmin, bbmax, bbinc])

CO_1 = widgets.HBox([CO_all, CO_035, CO_055, CO_070, CO_071])
CO_2 = widgets.HBox([CO_072, CO_073, CO_074, CO_075, CO_085])
CO_3 = widgets.HBox([CO_090, CO_091, CO_092, CO_093, CO_094])
CO_4 = widgets.HBox([CO_095, CO_100, CO_105, CO_112, CO_140])


CO = widgets.VBox([CO_1, CO_2, CO_3, CO_4])

met_1 = widgets.HBox([met_all, met__05, met_00])
met_2 = widgets.HBox([met_05, met_10, met_20])

met = widgets.VBox ([met_1, met_2])

Teff_1 = widgets.HBox([Teff_all, Teff_10, Teff_12, Teff_15])
Teff_2 = widgets.HBox([Teff_17, Teff_20, Teff_22, Teff_25])

Teff = widgets.VBox ([Teff_1, Teff_2])

SpT = widgets.HBox([spt_all, spt_f, spt_g, spt_k, spt_m])

logg = widgets.HBox([logg_all, logg_2, logg_3, logg_4, logg_5])


highlight1 = widgets.Box([hl1, c1])
highlight2 = widgets.Box([hl2, c2])
highlight3 = widgets.Box([hl3, c3])
highlight4 = widgets.Box([hl4, c4])
highlight5 = widgets.Box([hl5, c5])
highlight6 = widgets.Box([hl6, c6])
highlight7 = widgets.Box([hl7, c7])
highlight8 = widgets.Box([hl8, c8])
highlight9 = widgets.Box([hl9, c9])
highlight10 = widgets.Box([hl10, c10])


highlight = widgets.VBox([highlight1, highlight2, highlight3, highlight4, highlight5, highlight6, 
                          highlight7, highlight8, highlight9, highlight10])

accordion = widgets.Accordion(children=[bands, properties1, bboptions])
accordion.set_title(0, 'Photometric Bands')
accordion.set_title(1, 'Plot Properties')
accordion.set_title(2, 'Blackbody properties (optional)')

accordion1 = widgets.Accordion(children=[bands, properties, bboptions])
accordion1.set_title(0, 'Photometric Bands')
accordion1.set_title(1, 'Plot Properties')
accordion1.set_title(2, 'Blackbody properties (optional)')

accordion2 = widgets.Accordion(children=[bands, properties, bboptions, highlight])
accordion2.set_title(0, 'Photometric Bands')
accordion2.set_title(1, 'Plot Properties')
accordion2.set_title(2, 'Blackbody properties (optional)')
accordion2.set_title(3, 'Highlighted Planets (max 10)')

accordion4 = widgets.Accordion(children = [CO, met, SpT, Teff, logg])

accordion4.set_title(0, 'C/O Ratio')
accordion4.set_title(1, 'Metallicity')
accordion4.set_title(2, 'Host Star Spectral Type')
accordion4.set_title(3, 'Effective Temperature')
accordion4.set_title(4, 'Surface Gravity')

accordion3 = widgets.Accordion(children = [bands, properties2, bboptions, accordion4])
accordion3.set_title(0, 'Photometric Bands')
accordion3.set_title(1, 'Plot Properties')
accordion3.set_title(2, 'Blackbody properties (optional)')
accordion3.set_title(3, 'Model Spectra Options')

accordion5 = widgets.Accordion(children = [bands_synth, properties, bboptions])
accordion5.set_title(0, 'Photometric Bands')
accordion5.set_title(1, 'Plot Properties')
accordion5.set_title(2, 'Blackbody properties (optional)')


#tab_contents = [accordion, 'p1']
children = [accordion, accordion1, accordion2, accordion3, accordion5]
tab = widgets.Tab()
tab.children = children
tab.set_title(0, 'Style 1')
tab.set_title(1, 'Style 2')
tab.set_title(2, 'Style 3')
tab.set_title(3, 'Model Atmospheres')
tab.set_title(4, 'Synthetic Magnitudes')

out = widgets.Output(layout={'border': '1px solid black'})


button = widgets.Button(
    description='Plot Diagram',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check' # (FontAwesome names without the `fa-` prefix)
)

def on_button_clicked(b):
    COs = []
    if CO_035.value == True:
        COs.append(0.35)
    if CO_055.value == True:
        COs.append(0.55)
    if CO_070.value == True:
        COs.append(0.70)
    if CO_071.value == True:
        COs.append(0.71)
    if CO_072.value == True:
        COs.append(0.72)
    if CO_073.value == True:
        COs.append(0.73)
    if CO_074.value == True:
        COs.append(0.74)
    if CO_075.value == True:
        COs.append(0.75)
    if CO_085.value == True:
        COs.append(0.85)
    if CO_090.value == True:
        COs.append(0.90)
    if CO_091.value == True:
        COs.append(0.91)
    if CO_092.value == True:
        COs.append(0.92)
    if CO_093.value == True:
        COs.append(0.93)
    if CO_094.value == True:
        COs.append(0.94)
    if CO_095.value == True:
        COs.append(0.95)
    if CO_100.value == True:
        COs.append(1.00)
    if CO_105.value == True:
        COs.append(1.05)
    if CO_112.value == True:
        COs.append(1.12)
    if CO_140.value == True:
        COs.append(1.40)
    if CO_all.value == True:
        COs = [0.35, 0.55, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.85, 0.90, 0.91, 0.92, 
          0.93, 0.94, 0.95, 1.00, 1.05, 1.12, 1.40]

    FeHs = []
    if met__05.value == True:
        FeHs.append(-0.5)
    if met_00.value == True:
        FeHs.append(0)
    if met_05.value == True:
        FeHs.append(0.5)
    if met_10.value == True:
        FeHs.append(1)
    if met_20.value == True:
        FeHs.append(2)
    if met_all.value == True:
        FeHs = [-0.5, 0.0, 0.5, 1.0, 2.0]

    Teffs = []
    if Teff_10.value == True:
        Teffs.append(1000)
    if Teff_12.value == True:
        Teffs.append(1250)
    if Teff_15.value == True:
        Teffs.append(1500)
    if Teff_17.value == True:
        Teffs.append(1750)
    if Teff_20.value == True:
        Teffs.append(2000)
    if Teff_22.value == True:
        Teffs.append(2250)
    if Teff_25.value == True:
        Teffs.append(2500)
    if Teff_all.value == True:
        Teffs = [1000, 1250, 1500, 1750, 2000, 2250, 2500]

    SpTs = []
    if spt_f.value == True:
        SpTs.append('F5')
    if spt_g.value == True:
        SpTs.append('G5')
    if spt_k.value == True:
        SpTs.append('K5')
    if spt_m.value == True:
        SpTs.append('M5')
    if spt_all.value == True:
        SpTs = ['F5', 'G5', 'K5', 'M5']

    loggs = []
    if logg_2.value == True:
        loggs.append(2.3)
    if logg_3.value == True:
        loggs.append(3)
    if logg_4.value == True:
        loggs.append(4)
    if logg_5.value == True:
        loggs.append(5)
    if logg_all.value == True:
        loggs = [2.3, 3.0, 4.0, 5.0]
    
    with out:
        colour = str(band1.value+"-"+band2.value)
        mag = absmag.value
        colour_synth = str(band1_synth.value+"-"+band2_synth.value)
        mag_synth = absmag_synth.value
        
        
        if colourbar.value == True:
            fig, ax = plt.subplots(1, 1, figsize =(8, 9.5))
        else:
            fig, ax = plt.subplots(1, 1, figsize =(8, 8))
        if tab.selected_index == 0:
            ExoCMD.ExoCMD_1 (planets_path.value, BD_path.value, ax, colour, mag, adjusted = adjusted.value,
                       colourbar = colourbar.value, bb09 = bb09.value, 
                      bbmin=bbmin.value, bbmax=bbmax.value, bbinc=bbinc.value, polynomial=poly.value)
        elif tab.selected_index == 1:
            ExoCMD.ExoCMD_2 (planets_path.value, BD_path.value, ax, colour, mag, adjusted = adjusted.value,
                       colourbar = colourbar.value, bb09 = bb09.value, 
                      bbmin=bbmin.value, bbmax=bbmax.value, bbinc=bbinc.value)
        elif tab.selected_index == 2:
            ExoCMD.ExoCMD_3 (planets_path.value, BD_path.value, ax, colour, mag, adjusted = adjusted.value,
                       colourbar = colourbar.value, bb09 = bb09.value, 
                      bbmin=bbmin.value, bbmax=bbmax.value, bbinc=bbinc.value, highlight=[[hl1.value.upper(), c1.value]
                                                                                        , [hl2.value.upper(), c2.value]
                                                                                         , [hl3.value.upper(), c3.value]
                                                                                         , [hl4.value.upper(), c4.value]
                                                                                         , [hl5.value.upper(), c5.value]
                                                                                         , [hl6.value.upper(), c6.value]
                                                                                         , [hl7.value.upper(), c7.value]
                                                                                         , [hl8.value.upper(), c8.value]
                                                                                         , [hl9.value.upper(), c9.value]
                                                                                         , [hl10.value.upper(), c10.value]])
        elif tab.selected_index == 3:
            fig.set_size_inches(8, 9.5, forward = True)
            ExoCMD.ExoCMD_model (planets_path.value, BD_path.value, ax, colour, mag, adjusted = adjusted.value, SpT=SpTs, FeH=FeHs,
                           logg=loggs, CO=COs, Teff=Teffs, colour_by = colour_by.value)
            
        elif tab.selected_index == 4:
            ExoCMD.ExoCMD_synth(planets_path.value, ax, colour_synth, mag_synth, adjusted = adjusted.value,
                          colourbar = colourbar.value, bb09 = bb09.value, bbmin = bbmin.value, bbmax = bbmax.value,
                         bbinc = bbinc.value, bb18 = False, synth_file = 'synth_mags.txt')

button.on_click(on_button_clicked)

def run_all(ev):
    display(Javascript('IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+0, IPython.notebook.ncells())'))

button1 = widgets.Button(description="Clear All")
button1.on_click(run_all)

