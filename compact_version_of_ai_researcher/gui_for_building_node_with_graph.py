import warnings
warnings.filterwarnings("ignore")

from helper import ewriter, writer_gui

MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()
