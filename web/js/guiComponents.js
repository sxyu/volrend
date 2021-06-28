let guiInit = function() {
    let slider_bgbrightness = $('#slider-bgbrightness');
    slider_bgbrightness.val(1.0);
    slider_bgbrightness.on("input", function() {
        let opt = Volrend.get_options();
        opt.background_brightness = parseFloat(this.value);
        Volrend.set_options(opt);
    });

    let sliders_min = $('.slider-min');
    sliders_min.val(0.0);
    sliders_min.on("input", function() {
        let x = parseFloat(document.getElementById('slider-bbox-minx').value);
        let y = parseFloat(document.getElementById('slider-bbox-miny').value);
        let z = parseFloat(document.getElementById('slider-bbox-minz').value);

        let opt = Volrend.get_options();
        opt.render_bbox[0] = x;
        opt.render_bbox[1] = y;
        opt.render_bbox[2] = z;
        Volrend.set_options(opt);
    });
    let sliders_max = $('.slider-max');
    sliders_max.val(1.0);
    sliders_max.on("input", function() {
        let x = parseFloat(document.getElementById('slider-bbox-maxx').value);
        let y = parseFloat(document.getElementById('slider-bbox-maxy').value);
        let z = parseFloat(document.getElementById('slider-bbox-maxz').value);

        let opt = Volrend.get_options();
        opt.render_bbox[3] = x;
        opt.render_bbox[4] = y;
        opt.render_bbox[5] = z;
        Volrend.set_options(opt);
    });

    let sliders_decomp = $('.slider-decomp');
    sliders_decomp.on("input", function() {
        let vmin = parseInt(document.getElementById('slider-decomp-min').value);
        let vmax = parseInt(document.getElementById('slider-decomp-max').value);

        let opt = Volrend.get_options();
        opt.basis_minmax[0] = vmin;
        opt.basis_minmax[1] = vmax;
        Volrend.set_options(opt);
        document.getElementById('decomp-min-disp').innerText = vmin;
        document.getElementById('decomp-max-disp').innerText = vmax;
    });

    let sliders_vdir = $('.slider-vdir');
    sliders_vdir.val(0.0);
    sliders_vdir.on("input", function() {
        let vx = parseFloat(document.getElementById('slider-vdir-x').value);
        let vy = parseFloat(document.getElementById('slider-vdir-y').value);
        let vz = parseFloat(document.getElementById('slider-vdir-z').value);

        let opt = Volrend.get_options();
        opt.rot_dirs[0] = vx;
        opt.rot_dirs[1] = vy;
        opt.rot_dirs[2] = vz;
        Volrend.set_options(opt);
    });

    $('#options-close').click(function() {
        $('#options').css('display', 'none');
    });

    $('#options-btn').click(function() {
        $('#options').css('display', 'block');
    });

    $('#vdir-reset-btn').click(function() {
        sliders_vdir.val(0.0);
        let opt = Volrend.get_options();
        opt.rot_dirs[0] = 0.0;
        opt.rot_dirs[1] = 0.0;
        opt.rot_dirs[2] = 0.0;
        Volrend.set_options(opt);
    });

    $('#qual-radio-1').attr('checked', 'checked');
    $('input[name=qual-radio]').on('change', function() {
        let checked_val = $('input[name=qual-radio]:checked').val()
        let opt = Volrend.get_options();
        if (checked_val == 0) {
            opt.step_size = 2e-3;
            opt.stop_thresh = 1e-1;
            opt.sigma_thresh = 1e-1;
        } else if (checked_val == 1) {
            opt.step_size = 1e-3;
            opt.stop_thresh = 2e-2;
            opt.sigma_thresh = 2e-2;
        } else {
            opt.step_size = 1e-4;
            opt.stop_thresh = 1e-2;
            opt.sigma_thresh = 1e-2;
        }
        Volrend.set_options(opt);
    });

    let checkbox_showgrid = $('#checkbox-showgrid');
    checkbox_showgrid[0].checked = false;
    checkbox_showgrid.on('change', function() {
        let checked = $('#checkbox-showgrid')[0].checked == true;
        let opt = Volrend.get_options();
        opt.show_grid = checked;
        Volrend.set_options(opt);
    });

    let sliders_grid_reso = $('#slider-grid-reso');
    sliders_grid_reso.on("input", function() {
        let value = parseInt(document.getElementById('slider-grid-reso').value);
        let opt = Volrend.get_options();
        opt.grid_max_depth = value;
        Volrend.set_options(opt);
        document.getElementById('grid-reso-disp').innerText = value;
    });

    $('#mesh-add-cube-btn').click(function() {
        Volrend.mesh_add_cube([0.0, 0.0, 1.0], 0.2);
    });
    $('#mesh-add-sphere-btn').click(function() {
        Volrend.mesh_add_sphere([0.4, 0.0, 1.0], 0.1);
    });
};

let guiLoadTreeUpdate = function() {
    let sliders_decomp = $('.slider-decomp');
    sliders_decomp.attr('max', Volrend.get_basis_dim() - 1);
    $('#slider-decomp-min').val(0);
    $('#slider-decomp-max').val(Volrend.get_basis_dim() - 1);
    document.getElementById('decomp-min-disp').innerText = '0';
    document.getElementById('decomp-max-disp').innerText = Volrend.get_basis_dim() - 1;
};
