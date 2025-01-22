# Plot ICESat-2 ATL03 Photon data alongside Sentinel-2 

Run the notebook [IS2S2_plot_example.ipynb](IS2S2_plot_example.ipynb) to make your plots. 

![teaser image](https://raw.githubusercontent.com/fliphilipp/images/main/IS2_cycle09_RGT0842_GT2L_2020-11-18T08_20_16Z_strong.jpg)

The notebook uses the function ```plotIS2S2()```. 

Call it with the following arguments (bold ones are required):
- **```lat``` (required): latitude of your point of interest**
- **```lon``` (required): longitude of your point of interest**
- **```date``` (required): date in format 'YYYY-mm-dd'**
- **```rgt``` (required): the ICESat-2 track number**
- **```gtx``` (required): the ICESat-2 ground track (e.g. 'gt1l')**
- ```buffer_m```: the buffer in meters around the point of interest (default: 2500)
- ```ylim```: the y-axis limit of the plot (default: the matplotlib automatic limit)
- ```apply_geoid```: whether to apply geoid correction to photon heights (default: True)
- ```title```: the title for the plot (default: 'ICESat-2 ATL03 data')
- ```max_cloud_prob```: the maximum cloud probability within the area of interest in percent (default: 15)
- ```gamma_value```: the gamma value for image display (default: 1.0)
- ```inset```: whether to add inset map, so far either False or 'antarctica' (default: False)
- ```return_data```: whether to return the underlying ATL03 data and the imagery (default: False)
- ```re_download```: whether to re-download data that's already been downloaded (default: True)
- ```IS2dir```: the folder to which to save ATL03 granules (default: 'IS2data')
- ```imagery_filename```: the output path for the imagery (default: 'imagery/\<granule>_\<gtx>.tif')
- ```plot_filename```: the output path for the generated plot (default: 'plots/IS2_cycle\<cycle#\>\_RGT\<rgt>_\<gtx>\_\<date/time>\_\<beam_strength\>\_\<lon\>\_\<lat\>\_\<buffer_m\>m.jpg')

The function returns:
- **```figure```: matplotlib output figure** *(if ```return_data``` is set to ```False```)*
- ```(figure, atl03_dataframe, atl03_ancillary_data, imagery_rasterio_reader)``` *(if ```return_data``` is set to ```True```)*

*The examples in the notebook show how to plot and contextualize ICESat-2 signals of submerged "benches" at ice shelf fronts, which cause buoyant upward flexure at the front.*

By Philipp Arndt \
Scripps Institution of Oceanography, University of California San Diego \
Github: [@fliphilipp](https://github.com/fliphilipp) \
Contact: parndt@ucsd.edu
