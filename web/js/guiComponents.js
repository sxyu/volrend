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
};

let guiLoadTreeUpdate = function() {
    let sliders_decomp = $('.slider-decomp');
    sliders_decomp.attr('max', Volrend.get_basis_dim() - 1);
    $('#slider-decomp-min').val(0);
    $('#slider-decomp-max').val(Volrend.get_basis_dim() - 1);
    document.getElementById('decomp-min-disp').innerText = '0';
    document.getElementById('decomp-max-disp').innerText = Volrend.get_basis_dim() - 1;
};
