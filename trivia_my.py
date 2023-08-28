import numpy as np
import pandas as pd
import cmasher as cmr
import plotly.express as px
import plotly.graph_objects as go 
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from gofish import imagecube
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import PIL  # added by Aashish for rotating GIFs
import io   # added by Aashish for rotating GIFs

def read_cube(path, clip=None, rmin=None, rmax=None, N=None, vmin=None, vmax=None, dv=None, vunit_per_s=None):
    # Read in the FITS data.
    cube = imagecube(path)
    cube.data = cube.data.astype(float)
    if vunit_per_s == 'km':  # added by me
        cube.velax = cube.velax*1e3

    # Crop the data along the velocity axis, implemented from gofish
    vmin = cube.velax[0] if vmin is None else vmin*1.0e3
    vmax = cube.velax[-1] if vmax is None else vmax*1.0e3
#     vmin = 0.5*(cube.velax.min() + cube.velax.max()) - 5.0e3 if vmin is None else vmin*1.0e3
#     vmax = 0.5*(cube.velax.min() + cube.velax.max()) + 5.0e3 if vmax is None else vmax*1.0e3
    i = np.abs(cube.velax - vmin).argmin()
    i += 1 if cube.velax[i] < vmin else 0
    j = np.abs(cube.velax - vmax).argmin()
    j -= 1 if cube.velax[j] > vmax else 0
    cube.velax = cube.velax[i:j+1]
    cube.data = cube.data[i:j+1]
    
    if dv is not None:
        newvelax = np.arange(cube.velax[0], cube.velax[-1], dv*1.0e3)
        cs = CubicSpline(cube.velax, cube.data, axis=0)
        cube.data = cs(newvelax)
        cube.velax = newvelax

    # Generate a SNR mask
    mask_SNR = cube.data > clip * cube.rms
    
    # Generate a radial mask
    r, t, z = cube.disk_coords()
    rmin = 0 if rmin is None else rmin
    #     rmax = cube.FOV/3. if rmax is None else rmax
    rmax = cube.FOV*3 if rmax is None else rmax  # Added by Aashish to be conservative
    mask_r = np.logical_and(r >= rmin, r <= rmax)
    mask_r = np.tile(mask_r, (cube.data.shape[0], 1, 1))

    # Generate a combined mask
    mask = np.logical_and(mask_SNR, mask_r)

    # Masked LOS velocity, RA, Dec, intensity arrays.
    v = np.around((cube.velax[:, None, None] * np.ones(cube.data.shape))[mask]/1e3,decimals=3)
    x = np.around((cube.xaxis[None, None, :] * np.ones(cube.data.shape))[mask],decimals=3)
    y = np.around((cube.yaxis[None, :, None] * np.ones(cube.data.shape))[mask],decimals=3)
    i = np.around(cube.data[mask],decimals=3)

    # Take N random voxel.
    N = np.int(np.max([v.size/1.0e5,1])) if N is None else N
    if N > 1:
        idx = np.arange(v.size) 
        np.random.shuffle(idx)
        v = v[idx][::N]
        x = x[idx][::N]
        y = y[idx][::N]
        i = i[idx][::N]

    if (v.shape[0] > 1.0e6):
        print("Warning: There are total", v.shape[0], "points to present. The output file can be very large! Consider using a smaller N.")

    # Normalize the intensity.
    i = (i - i.min())/(i.max() - i.min())
#     print(vmin,vmax,min(v),max(v))

#     return(cube, x, y, v, vmin/1e3, vmax/1e3, i)
    return(cube, x, y, v, vmin, vmax, i)


def make_ppv(path, clip=3., rmin=None, rmax=None, N=None, cmin=None, cmax=None, constant_opacity=None, ntrace=20, 
        marker_size=2, cmap=None, hoverinfo='x+y+z', xaxis_title=None, 
        yaxis_title=None, zaxis_title=None, xaxis_backgroundcolor=None, xaxis_gridcolor=None,
        yaxis_backgroundcolor=None, yaxis_gridcolor=None,
        zaxis_backgroundcolor=None, zaxis_gridcolor=None,
        xmin=None, xmax=None, ymin=None, ymax=None, vmin=None, vmax=None, dv=None,
        projection_x=False, projection_y=False, projection_z=True,
        show_colorbar=True, camera_eye_x=-1., camera_eye_y=-2., camera_eye_z=1.,
        show_figure=False, write_pdf=False, write_png=False, write_html=True, write_csv=False,
        vunit_per_s='m', source_ra_off=None, source_dec_off=None, source_vel=None, bool_traj = False, traj = None,
        path2=None, clip2=3., marker_color=None,marker_color2=None,out_filename=None,
        write_gif=False,gif_start_ang=90,gif_duration=3,gif_N_angs = 15,gif_loops=0):
    """
    Make a three-dimensional position-position-velocity diagram.

    Args:
        path (str): Relative path to the FITS cube.
        clip (Optional[float]): Clip the cube having cube.data > clip * cube.rms
        rmin (Optional[float]): Inner radius of the radial mask 
        rmax (Optional[float]): Outer radius of the radial mask 
        N (Optional[integer]): Downsample the data by a factor of N. 
        cmin (Optional[float]): The lower bound of the velocity for the colorscale in km/s. 
        cmax (Optional[float]): The upper bound of the velocity for the colorscale in km/s. 
        constant_opacity (Optional[float]): If not None, use a constant opacity of the given value.
        ntrace (Optional[integer]): Number of opacity layers.
        markersize (Optional[integer]): Size of the marker in the PPV diagram.
        cmap (Optional[str]): Name of the colormap to use for the PPV diagram.
        hoverinfo (Optional[str]): Determines which trace information appear on hover.
                   Any combination of "x", "y", "z", "text", "name" joined with a "+" 
                   or "all" or "none" or "skip". If `none` or `skip` are set, no 
                   information is displayed upon hovering. But, if `none` is set, 
                   click and hover events are still fired.
        xaxis_title (Optional[str]): X-axis title.
        yaxis_title (Optional[str]): Y-axis title.
        zaxis_title (Optional[str]): Z-axis title.
        xaxis_backgroundcolor (Optional[str]): X-axis background color.
        xaxis_gridcolor (Optional[str]): X-axis grid color.
        yaxis_backgroundcolor (Optional[str]): Y-axis background color.
        yaxis_gridcolor (Optional[str]): Y-axis grid color.
        zaxis_backgroundcolor (Optional[str]): Z-axis background color.
        zaxis_gridcolor (Optional[str]): Z-axis grid color.
        xmin (Optional[float]): The lower bound of PPV diagram X range.
        xmax (Optional[float]): The upper bound of PPV diagram X range.
        ymin (Optional[float]): The lower bound of PPV diagram Y range.
        ymax (Optional[float]): The upper bound of PPV diagram Y range.
        vmin (Optional[float]): The lower bound of PPV diagram Z range in km/s.
        vmax (Optional[float]): The upper bound of PPV diagram Z range in km/s.
        dv (Optional[float]): Desired velocity resolution in km/s.
        projection_x (Optional[bool]): Whether or not to add projection on the Y-Z plane.
        projection_y (Optional[bool]): Whether or not to add projection on the X-Z plane.
        projection_z (Optional[bool]): Whether or not to add projection on the X-Y plane.
        show_colorbar (Optional[bool]): Whether or not to plot a colorbar.
        camera_eye_x (Optional[float]): The x component of the 'eye' camera vector.
        camera_eye_y (Optional[float]): The y component of the 'eye' camera vector.
        camera_eye_z (Optional[float]): The z component of the 'eye' camera vector.
        show_figure (Optional[bool]): If True, show PPV diagram.
        write_pdf (Optional[bool]): If True, write PPV diagram in a pdf file.
        write_png (Optional[bool]): If True, write PPV diagram in a png file.
        write_html (Optional[bool]): If True, write PPV diagram in a html file.
        write_csv (Optional[bool]): If True, write the data to create the PPV diagram in a csv file.
        vunit_per_s (Optional[str]): Unit of spectral axis is assumed m(per second), could be changed to km. Added by Aashish.
        source_ra_off (Optional[float]): R.A. offset (arcsec) of a YSO wrt cube centre for a special marker. Added by Aashish.
        source_dec_off (Optional[float]): Decl. offset (arcsec) of a YSO wrt cube centre for a special marker. Added by Aashish.
        source_vel (Optional[float]): Radial velocity of a YSO for a special marker. Added by Aashish.
        bool_traj (Optional[bool]): If True, display trajectory profile(s).
        traj (Optional[array]): Array (list of lists, 2D numpy array, etc.) of shape 3xn which gives trajectory of particle in R.A., Decl., and R.V. domain.
        path2 (Optional[str]): Path to secondary fits file
        clip2 (Optional[float]): Clip the second cube having cube2.data > clip * cube2.rms
        marker_color (Optional[str]): Color for all the cube data markers, overrides cmap
        marker_color2 (Optional[str]): Color for all the second cube data markers, overrides cmap
        out_filename (Optional[str]): Name for the output files (.html, .pdf and .gif)
        write_gif (Optional[bool]): If True, save a rotating plot as a gif
        gif_start_ang (Optional[float]): Angle (projection) (in degrees) for the starting and ending frame of the gif, can be experimented with
        gif_duration (Optional[float]): Total duration of one gif loop, in seconds
        gif_N_angs (Optional[float]): Total no. of frames in one gif loop
        gif_loops (Optional[int]): Total no. of time gif loops, by default it is 0 and it means gif doesn't stop looping
        
    Returns:
        PPV diagram. Can also save in a pdf, html or gif format.
    """
 
    vmin0 = vmin
    vmax0 = vmax
    cube, x, y, v, vmin, vmax, i = read_cube(path, clip=clip, rmin=rmin, rmax=rmax, N=N, vmin=vmin0, vmax=vmax0, dv=dv, vunit_per_s=vunit_per_s)
    
    # Determine the opacity of the data points.
    cuts = np.linspace(0, 1, ntrace+1)
    opacity = np.logspace(-1., 0.5, cuts.size - 1)
    if constant_opacity is not None:
        opacity[:] = constant_opacity
    data = []

    xaxis_title = 'R.A. offset [arcsec]' if xaxis_title is None else xaxis_title
    yaxis_title = 'Decl. offset [arcsec]' if yaxis_title is None else yaxis_title
    zaxis_title = 'Radial velocity [km/s]' if zaxis_title is None else zaxis_title
#     xaxis_backgroundcolor = 'white' if xaxis_backgroundcolor is None else xaxis_backgroundcolor
#     xaxis_gridcolor = 'gray' if xaxis_gridcolor is None else xaxis_gridcolor
#     yaxis_backgroundcolor = 'white' if yaxis_backgroundcolor is None else yaxis_backgroundcolor
#     yaxis_gridcolor = 'gray' if yaxis_gridcolor is None else yaxis_gridcolor
#     zaxis_backgroundcolor = 'white' if zaxis_backgroundcolor is None else zaxis_backgroundcolor
#     zaxis_gridcolor = 'gray' if zaxis_gridcolor is None else zaxis_gridcolor
    
    xmin, xmax, ymin, ymax = min(x),max(x),min(y),max(y) # added by Aashish
#     xmin = cube.FOV/2.0 if xmin is None else xmin
#     xmax = -cube.FOV/2.0 if xmax is None else xmax
#     ymin = -cube.FOV/2.0 if ymin is None else ymin
#     ymax = cube.FOV/2.0 if ymax is None else ymax
#     if rmax is not None:
#         xmin, xmax, ymin, ymax = rmax, -rmax, -rmax, rmax
    
    colorscale = make_colorscale('cmr.pride') if cmap is None else cmap
#     cmin = min(v)/1.0e3 if cmin is None else cmin # added by Aashish
#     cmax = max(v)/1.0e3 if cmax is None else cmax # added by Aashish
    cmin = vmin/1.0e3 if cmin is None else cmin
    cmax = vmax/1.0e3 if cmax is None else cmax

    # 3d scatter plot
    for a, alpha in enumerate(opacity):
        mask = np.logical_and(i >= cuts[a], i < cuts[a+1])
        if marker_color is None:
            data += [go.Scatter3d(x=x[mask], y=y[mask], z=v[mask], mode='markers',
                                   marker=dict(size=marker_size, color=v[mask], colorscale=colorscale,
                                               cauto=False, cmin=cmin, cmax=cmax,
                                               opacity=min(1.0, alpha)),
                                   hoverinfo=hoverinfo,name='Data', 
                                  )
                     ]
        else:
            data += [go.Scatter3d(x=x[mask], y=y[mask], z=v[mask], mode='markers',
                               marker=dict(size=marker_size, color=marker_color,
                                           opacity=min(1.0, alpha)),
                               hoverinfo=hoverinfo,name='Data', 
                              )
                 ]
    
    ## Plotting data from another cube
    data2 = []
    if path2 is not None:
        cube2, x2, y2, v2, vmin2, vmax2, i2 = read_cube(path2, clip=clip2, rmin=rmin, rmax=rmax, N=N, vmin=vmin0, vmax=vmax0, dv=dv, vunit_per_s=vunit_per_s)
        for a, alpha in enumerate(opacity):
            mask = np.logical_and(i2 >= cuts[a], i2 < cuts[a+1])
            if marker_color2 is None:
                data2 += [go.Scatter3d(x=x2[mask], y=y2[mask], z=v2[mask], mode='markers',
                                       marker=dict(size=marker_size, color=v2[mask], colorscale=colorscale,
                                                   cauto=False, cmin=cmin, cmax=cmax,
                                                   opacity=min(1.0, alpha)),
                                       hoverinfo=hoverinfo,name='Data 2', 
                                      )
                         ]
            else:
                data2 += [go.Scatter3d(x=x2[mask], y=y2[mask], z=v2[mask], mode='markers',
                                   marker=dict(size=marker_size, color=marker_color2,
                                               opacity=min(1.0, alpha)),
                                   hoverinfo=hoverinfo,name='Data 2', 
                                  )
                     ]

                
    ### Add a special marker for source
    source = []
    if source_ra_off is not None:
#         print(source_ra_off,source_dec_off,source_vel)
        source_vel = np.median(v) if source_vel is None else source_vel
        source += [go.Scatter3d(x=[source_ra_off], y=[source_dec_off], z=[source_vel],# mode='markers',
                                   marker=dict(size=10, color='black',
                                               opacity=0.9),
                               hoverinfo=hoverinfo,name='Source',)]

        
    ## overplotting trajectories
    traj_line = []
    if bool_traj:
        if traj != None:
            print("Showing Trajectory...")
            traj_line += [go.Scatter3d(x=traj[0], y=traj[1], z=traj[2], mode='lines', line=dict(color='black', width=8))] 
#            traj_line += [go.Scatter3d(x=traj[0], y=traj[1], z=traj[2], mode='markers+lines', line=dict(color='black', width=8), marker=dict(color='black', size=3))]  # just "mode='lines'" would work as well, added markers to show projections
        else:  # Some defualt profiles to plot... not sure what is the best default, can be deleted
            r = np.linspace(xmin, xmax, 51)
            v_sys = 7.5
            v_rot = 10
            v = v_sys + v_rot / np.sqrt(np.abs(r)) * np.sign(r)
            l_rot = []
            for theta in np.arange(-25,11,5):
                traj_line += [go.Scatter3d(x=traj[0], y=traj[1], z=traj[2], mode='lines', line=dict(color='black', width=8))] 
    #            traj_line += [go.Scatter3d(x=traj[0], y=traj[1], z=traj[2], mode='markers+lines', line=dict(color='black', width=8), marker=dict(color='black', size=3))]  # just "mode='lines'" would work as well, added markers to show projections
        
## ----------------------- copied from jonathan's code

#     ## velocity profiles
#     ##first try plotting lines (but see below - surface is better)
#     r = np.linspace(xmin, xmax, 51)
#     v_sys = 7.5
#     v_rot = 10
#     v = v_sys + v_rot / np.sqrt(np.abs(r)) * np.sign(r)
#     l_rot = []
#     for theta in np.arange(-25,11,5):
#         datas += [go.Scatter3d(x=r*np.cos(np.radians(theta)), y=r*np.sin(np.radians(theta)), z=v, mode='lines', line=dict(color='black', width=5))]
# #     fig = go.Figure(data = l_rot)

    ## plot surface, different angular extents on each side
#     v_sys = 3.9
#     v_rot = 6.2    # pure rotation
#     #v_rot = 3.6    # rotation for a 2 Msun central mass
#     v_fall = 20    # pure infall
#     v_fall = 12    # infall added on to 2 Msun rotation

#     r1 = np.linspace(xmin/20,xmin,21)
#     t1 = np.radians(np.linspace(-30,30,21))
#     RR, TT = np.meshgrid(r1, t1)
#     X1 = RR * np.cos(TT)
#     Y1 = RR * np.sin(TT)
#     Z1_rot = v_rot / np.sqrt(np.abs(RR)) * np.sign(RR)
#     Z1_fall = v_fall / RR

#     r2 = np.linspace(xmax/20,xmax,21)
#     t2 = np.radians(np.linspace(-30,10,21))
#     RR, TT = np.meshgrid(r2, t2)
#     X2 = RR * np.cos(TT)
#     Y2 = RR * np.sin(TT)
#     Z2_rot = v_rot / np.sqrt(np.abs(RR)) * np.sign(RR)
#     Z2_fall = v_fall / RR

#     #fig = go.Figure(data = s_red + s_green + s_blue + l_rot)
# #     fig = go.Figure(data = s_red + s_green + s_blue)

#     plot_rotation = True
#     plot_infall = True

#     if plot_rotation:
#         print('Plotting rotation profile')
#         fig.add_surface(x=X1, y=Y1, z=v_sys+Z1_rot, opacity=0.5, showscale=False, surfacecolor=0*Z1_rot, colorscale='gray')
#         fig.add_surface(x=X2, y=Y2, z=v_sys+Z2_rot, opacity=0.5, showscale=False, surfacecolor=0*Z2_rot, colorscale='gray')
#     if plot_infall:
#         print('Plotting infall profile')
#         fig.add_surface(x=X1, y=Y1, z=v_sys+Z1_fall, opacity=0.5, showscale=False, surfacecolor=0*Z1_fall, colorscale='gray')
#         fig.add_surface(x=X2, y=Y2, z=v_sys+Z2_fall, opacity=0.5, showscale=False, surfacecolor=0*Z2_fall, colorscale='gray')
#     if plot_infall and plot_rotation:
#         print('Plotting infall and rotation')
#         fig.add_surface(x=X1, y=Y1, z=v_sys+Z1_rot+Z1_fall, opacity=0.5, showscale=False, surfacecolor=0*Z1_fall, colorscale='plasma')
#         fig.add_surface(x=X2, y=Y2, z=v_sys+Z2_rot+Z2_fall, opacity=0.5, showscale=False, surfacecolor=0*Z2_fall, colorscale='plasma')

## -----------------------

    # layout
    layout = go.Layout(scene=dict(xaxis_title=xaxis_title, 
                                  yaxis_title=yaxis_title,
                                  zaxis_title=zaxis_title,
                                  xaxis_backgroundcolor=xaxis_backgroundcolor, 
                                  xaxis_gridcolor=xaxis_gridcolor,
                                  yaxis_backgroundcolor=yaxis_backgroundcolor, 
                                  yaxis_gridcolor=yaxis_gridcolor,
                                  zaxis_backgroundcolor=zaxis_backgroundcolor, 
                                  zaxis_gridcolor=zaxis_gridcolor,
                                  xaxis_range=[xmin, xmax],
                                  yaxis_range=[ymin, ymax],
                                  zaxis_range=[vmin/1.0e3, vmax/1.0e3],
                                  aspectmode='cube'),
#                       margin=dict(l=0, r=0, b=0, t=0), 
                       margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
                       )

    fig = go.Figure(data=data+data2+source+traj_line, layout=layout)

    proj_opacity = 0.9
    fig.update_traces(projection_x=dict(show=projection_x, opacity=proj_opacity), 
                      projection_y=dict(show=projection_y, opacity=proj_opacity),
                      projection_z=dict(show=projection_z, opacity=proj_opacity),
                     )

    if show_colorbar:
        fig.update_traces(marker_colorbar=dict(thickness=20, 
#                                               tickvals=np.arange(cmin,cmax+1),
                                               tickformat='.1f',
                                               title='v [km/s]',
                                               title_side='right',
                                               len=0.5
                                              )
                         )
#        fig.update_layout(coloraxis_colorbar_x=-1.)   

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=camera_eye_x, y=camera_eye_y, z=camera_eye_z)
                 )

    fig.update_layout(scene_camera=camera)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    out_filename = path.replace('.fits', '_ppv') if out_filename is None else out_filename
    
    if show_figure:
        fig.show()
    
    if write_pdf:
        fig.write_image(out_filename +'.pdf',scale=3)
    
    if write_png:
        fig.write_image(out_filename +'.png',scale=3)
    
    if write_html:
        fig.write_html(out_filename +'.html', include_plotlyjs=True)
    
    if write_gif:
        print("Saving GIF...")
        
        # Rotate the plot
        ang_step = 360/gif_N_angs   # Step between angles
        angs1 = np.arange(gif_start_ang,gif_start_ang-180,-ang_step)
        angs2 = np.arange(gif_start_ang,gif_start_ang+180,ang_step)
        angs = np.concatenate([angs2,(angs1[::-1]+360)%360])   # Array of all angles
        zoom_fac = 2.3 # Adjusts size of the image
        frames = []
        for i in range(gif_N_angs):
            frame = go.Frame(layout=dict(scene_camera=dict(eye=dict(x=zoom_fac*np.cos(np.radians(angs[i])),
                                                             y=zoom_fac*np.sin(np.radians(angs[i])),
                                                             z=zoom_fac*0.5))))
            frames.append(frame)
        fig.frames = frames

#         ### Following is to show the rotating plot in jupyter, not required for GIF
#         # Set animation settings
#         animation_settings = dict(frame=dict(duration=50, redraw=True),romcurrent=True,
#             transition=dict(duration=100, easing='quadratic-in-out'),)
#         # Add buttons to control the animation
#         fig.update_layout(updatemenus=[dict(type="buttons",buttons=[
#                      dict(label="Play",method="animate",args=[None, animation_settings]),
#                      dict(label="Pause",method="animate",args=['null',dict(mode= "immediate")])
#                  ],),])

        # generate images for each step in animation
        gif_frames = []
        for s, fr in enumerate(fig.frames):
            # set main traces to appropriate traces within plotly frame
            fig.update(data=fr.data,layout=fr.layout)
            # generate image of current state
            gif_frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png"))))

        # create animated GIF
        gif_frames[0].save(
                out_filename +'.gif',
                save_all=True,
                append_images=gif_frames[1:],
                optimize=True,
                duration=(gif_duration*1000)/gif_N_angs,  # Total will take gif_duration*1000 milli-secs, /gif_N_angs for per frame
                loop=gif_loops,
            )

    if write_csv:
        df = pd.DataFrame({"RA offset" : x, "Dec offset" : y, "velocity" : v})
        df.to_csv(out_filename +'.csv', float_format='%.3e', index=False)
    return


def concatenate_cmaps(cmap1, cmap2, ratio=None, ntot=None):
    """
    Concatenate two colormaps.
    https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html

    Args:
        cmap1 (str): Name of the first colormap (bottom) to concatenate.
        cmap2 (str): Name of the second colormap (top) to concatenate.
        ratio (Optional[float]): The ratio between the first and second colormap.
        ntot (Optional[int]): The number of levels in the concatenated colormap.
    """
    ratio = 0.5 if ratio is None else ratio
    ntot = 256 if ntot is None else ntot

    bottom = cm.get_cmap(cmap1, ntot)
    top = cm.get_cmap(cmap2, ntot)
    nbottom = int(ratio*ntot)
    ntop = ntot-nbottom
    newcolors = np.vstack((bottom(np.linspace(0, 1, nbottom)),
                       top(np.linspace(0, 1, ntop))))
    newcmp = ListedColormap(newcolors, name='newcolormap')    
    newcmp = np.around(newcmp(range(ntot)),decimals=4)
    colorscale = [[f, 'rgb({}, {}, {})'.format(*newcmp[ff])]
              for ff, f in enumerate(np.around(np.linspace(0, 1, newcmp.shape[0]),decimals=4))]
    return colorscale

def make_colorscale(cmap):
    """
    Convert a color table into a CSS-compatible color table.

    Args:
        cmap (str): Color table name. e.g., 'cmr.pride'

    Returns:
        A list containing CSS-compatible color table.
    """
    cmarr = np.array(cmr.take_cmap_colors('cmr.pride', 128))
    colorscale = [[f, 'rgb({}, {}, {})'.format(*cmarr[ff])]
                  for ff, f in enumerate(np.linspace(0, 1, cmarr.shape[0]))]
    return colorscale

def make_cm(path, clip=3., fmin=None, fmed=None, fmax=None, vmin=None, vmax=None,
            xmin=None, xmax=None, ymin=None, ymax=None,
            nx=None, ny=None, cmap=None, nointerp=False, show_figure=False, write_html=True):
    """
    Make interactive channel map.

    Args:
        path (str): Relative path to the FITS cube.
        clip (Optional[float]): Plot cube.data < clip * cube.rms in black and white.
        fmin (Optional[float]): The lower bound of the flux. 
        fmed (Optional[float]): The boundary between bw/color cmaps.
        fmax (Optional[float]): The upper bound of the flux.
        vmin (Optional[float]): The lower bound of the velocity in km/s. 
        vmax (Optional[float]): The upper bound of the velocity in km/s.
        xmin (Optional[float]): The lower bound of X range.
        xmax (Optional[float]): The upper bound of X range.
        ymin (Optional[float]): The lower bound of Y range.
        ymax (Optional[float]): The upper bound of Y range.
        nx (Optional[float]): Number of x pixels.
        ny (Optional[float]): Number of y pixels.
        cmap (Optional[str]): Color map to use.
        nointerp (Optional[bool]): If True, no interpolation applied to the data.
        show_figure (Optional[bool]): If True, show channel map.
        write_html (Optional[bool]): If True, write channel map in a html file.
    Returns:
        Interactive channel map in a html format.
    """
    # Read in the FITS data.
    cube = imagecube(path)
    cube.data = cube.data.astype(float)

    fmin = 0. if fmin is None else fmin
    fmed = clip*cube.rms if fmed is None else fmed
    fmax = cube.data.max()*0.7 if fmax is None else fmax
    funit = 'Jy/beam'
    if fmax < 0.5 :
        cube.data *= 1.0e3
        fmin *= 1.0e3
        fmed *= 1.0e3
        fmax *= 1.0e3
        funit = 'mJy/beam'

    if xmin is None:
        xmin = cube.FOV/2.0 
        i = -1
    else:
        xmin = xmin
        i = np.abs(cube.xaxis - xmin).argmin()
        i += 1 if cube.xaxis[i] < xmin else 0
    if xmax is None:
        xmax = -cube.FOV/2.0
        j = -1
    else:
        xmax = xmax
        j = np.abs(cube.xaxis - xmax).argmin()
        j -= 1 if cube.xaxis[j] > xmax else 0

    cube.xaxis = cube.xaxis[j+1:i]
    cube.data = cube.data[:,:,j+1:i]

    if ymin is None:
        ymin = -cube.FOV/2.0
        i = 0
    else:
        ymin = ymin
        i = np.abs(cube.yaxis - ymin).argmin()
        i += 1 if cube.yaxis[i] < ymin else 0
    if ymax is None:
        ymax = cube.FOV/2.0
        j = -1
    else:
        ymax = ymax
        j = np.abs(cube.yaxis - ymax).argmin()
        j -= 1 if cube.yaxis[j] > ymax else 0

    cube.yaxis = cube.yaxis[i:j]
    cube.data = cube.data[:,i:j,:]

    # Crop the data along the velocity axis, implemented from gofish
    vmin = cube.velax[0] if vmin is None else vmin*1.0e3
    vmax = cube.velax[-1] if vmax is None else vmax*1.0e3
    i = np.abs(cube.velax - vmin).argmin()
    i += 1 if cube.velax[i] < vmin else 0
    j = np.abs(cube.velax - vmax).argmin()
    j -= 1 if cube.velax[j] > vmax else 0
    cube.velax = cube.velax[i:j+1]
    cube.data = cube.data[i:j+1]

    if (cube.velax.shape[0] > 200.):
        print("Warning: There are total", cube.velax.shape[0], "channels. The output file can be very large! Consider using a smaller velocity range by changing vmin and vmax.")

    # Interpolate the cube on the RA-Dec plane
    # Caution: This is only for visualization purposes.
    # Avoid using this interpolation routine for scientific purposes. 
    if not nointerp:
        nx = 400 if nx is None else nx
        ny = 400 if ny is None else ny

        oldx = cube.xaxis
        oldy = cube.yaxis

        cube.xaxis = np.linspace(cube.xaxis[0],cube.xaxis[-1],nx)
        cube.yaxis = np.linspace(cube.yaxis[0],cube.yaxis[-1],ny)
        cube.nxpix, cube.nypix = nx, ny

        newx, newy = np.meshgrid(cube.xaxis, cube.yaxis)
        newdata = np.zeros((cube.data.shape[0],ny,nx))

        for i in np.arange(cube.data.shape[0]):
            interp_func = RegularGridInterpolator((oldy, oldx[::-1]), cube.data[i])
            newdata[i] = interp_func(np.array([newy.flatten(), newx.flatten()]).T).reshape((ny,nx))[:,::-1]
        cube.data = newdata
    else:
        print("Warning: No interpolation will perform. The output file can be very large!")

    cube.xaxis = np.around(cube.xaxis,decimals=3)
    cube.yaxis = np.around(cube.yaxis,decimals=3)

    toplot = np.around(cube.data,decimals=3)

    cmap = concatenate_cmaps('binary','inferno',ratio=fmed/fmax) if cmap is None else concatenate_cmaps('binary',cmap,ratio=fmed/fmax)

    fig = px.imshow(toplot, color_continuous_scale=cmap, origin='lower', 
                    x=cube.xaxis, y=cube.yaxis, 
                    zmin=fmin, zmax=fmax, 
                    labels=dict(x="RA offset [arcsec]", y="Dec offset [arcsec]", 
                                color="Intensity ["+funit+"]", animation_frame="channel"),
                    animation_frame=0,
                   )
#    fig.update_xaxes(range=[xmin, xmax],autorange=False)
#    fig.update_yaxes(range=[ymax, ymin],autorange=False)
    fig.update_xaxes(autorange="reversed")
    fig.update_xaxes(ticks="outside")
    fig.update_yaxes(ticks="outside")
    for i, frame in enumerate(fig.frames):
        frame['layout'].update(title_text="v = {:.2f} km/s".format(cube.velax[i]/1.0e3),
                               title_x=0.5,
                              ) 
    
    if show_figure:
       fig.show()
    if write_html:
       fig.write_html(path.replace('.fits', '_channel.html'), include_plotlyjs='cdn')
    return
