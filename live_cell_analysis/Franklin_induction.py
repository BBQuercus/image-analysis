time_points = [20, 20, 100]
laser_intensities = [20, 100, 50]

for i in range(3):
    VV.Acquire.TimeLapse.TimePoints = time_points[i]
    VV.Illumination.SetComponentSlider('Toptica-488_Laser488', laser_intensities[i])
    VV.Acquire.Sequence.Start()
    VV.Macro.Contro.WaitFor('VV.Acquire.IsRunning', '==', 'False')