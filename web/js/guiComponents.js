let populateLayers = function() {
    let layers_list = $('#layers-items');
    let html = "";
    let template = $('#layers-item-template').html();
    let template_collection = $('#layers-collection-template').html();
    let re_name = new RegExp('{name}', 'g');
    let re_space = new RegExp(' ', 'g');
    let re_full_name = new RegExp('{full_name}', 'g');
    let re_classes = new RegExp('{classes}', 'g');
    let re_bg_color = new RegExp('{bg_color}', 'g');
    let re_border_color = new RegExp('{border_color}', 'g');
    let re_mleft = new RegExp('{mleft}', 'g');
    let re_id = new RegExp('{id}', 'g');
    const invis_class = "layers-item-invisible";

    let mesh_cnt = Volrend.mesh_count();
    layer_tree = {__leaf : false};
    for (let i = 0; i < mesh_cnt; i++) {
        let mesh_name = Volrend.mesh_get_name(i);
        if (mesh_name.length == 0 || mesh_name[0] == '_') {
            // Hide from legend
            continue;
        }
        let mesh_color = Volrend.mesh_get_color(i);
        let mesh_is_visible = Volrend.mesh_get_visible(i);
        let mesh_color_str = "rgba(" + mesh_color[0] * 255.0 + ","
                                     + mesh_color[1] * 255.0 + ","
                                     + mesh_color[2] * 255.0 + ")";
        let mesh_bg_color_str = mesh_is_visible ? mesh_color_str : "#fff";
        let classes_str = mesh_is_visible ? "" : " " + invis_class;

        let mesh_name_tree = mesh_name.split('/');
        
        let it = layer_tree;

        for (let i = 0; i < mesh_name_tree.length - 1; i++) {
            let name_part = mesh_name_tree[i];
            if (!(name_part in it)) {
                it[name_part] = {
                    __leaf : false,
                    __name : name_part
                };
            }
            it = it[name_part];
        }
        let last_name = mesh_name_tree[mesh_name_tree.length-1];
        it[last_name] = {
            __leaf : true,
            __name : last_name,
            __full_name : mesh_name,
            color : mesh_color,
            is_visible : mesh_is_visible,
            color_str : mesh_color_str,
            bg_color_str : mesh_bg_color_str,
            classes_str : classes_str,
            mesh_id : i
        };

    }

    let dfs_populate_html = function(obj, full_name, depth) {
        if (obj.__leaf) {
            html += template.replace(re_name, obj.__name)
                .replace(re_full_name, obj.full_name)
                .replace(re_id, obj.mesh_id)
                .replace(re_bg_color, obj.bg_color_str)
                .replace(re_border_color, obj.color_str)
                .replace(re_classes, obj.classes_str)
                .replace(re_mleft, (depth - 1) + 'em');
        } else {
            if ('__name' in obj) {
                html += template_collection.replace(re_name, obj.__name)
                    .replace(re_id, obj.__name.replace(re_space, '-'))
                    .replace(re_full_name, full_name)
                    .replace(re_classes, "")
                    .replace(re_mleft, (depth - 1) + 'em');
            }
            for (var key in obj) {
                if (!key.startsWith('__')) {
                    dfs_populate_html(obj[key], full_name + '__' + key, depth + 1);
                }
            }
            if ('__name' in obj) {
                html += "\n</div>\n";
            }
         }
    };

    let dfs_visible_all = function(obj, new_visible) {
        if (obj.__leaf) {
            Volrend.mesh_set_visible(obj.mesh_id, new_visible);
            let base_ele = $("#layer-item-" + obj.mesh_id);
            let color_ele = $("#layer-item-color-" + obj.mesh_id);
            if (new_visible) {
                base_ele.removeClass(invis_class);
                color_ele.css("background-color", color_ele.attr("layer-color"));
            } else {
                base_ele.addClass(invis_class);
                color_ele.css("background-color", "#fff");
            }
        } else {
            for (var key in obj) {
                if (!key.startsWith('__')) {
                    dfs_visible_all(obj[key], new_visible);
                }
            }
         }
    };

    dfs_populate_html(layer_tree, "", 0);
    if (mesh_cnt > 0) {
        $('#layers-btn').show();
    }

    layers_list.html(html);
    $('.layers-item').click(function() {
        let $this = $(this);
        let color_ele = $this.children('.layers-item-color');
        let layer_id = parseInt($this.attr("layer"));
        let new_visible = !Volrend.mesh_get_visible(layer_id);
        Volrend.mesh_set_visible(layer_id, new_visible);
        if (new_visible) {
            $this.removeClass(invis_class);
            color_ele.css("background-color", color_ele.attr("layer-color"));
        } else {
            $this.addClass(invis_class);
            color_ele.css("background-color", "#fff");
        }
    });

    $('.layers-collection-action').click(function() {
        let $this = $(this);
        let collid = $this.parent().attr("collection-id");
        let collid_path = collid.slice(2).split('__')
        let curr = layer_tree;
        for (var i = 0; i < collid_path.length; i++) {
            curr = curr[collid_path[i]];
        }
        dfs_visible_all(curr, $this.hasClass('layers-collection-action-all'));
    });

    // Update time
    let slider_time = $('#slider-time');
    let slider_time_label_curr = $('#slider-time-label-curr');
    let slider_time_label_max = $('#slider-time-label-max');
    let curr_time = Volrend.get_time();
    let max_time = Volrend.mesh_max_time();
    slider_time_label_curr.text(curr_time);
    slider_time_label_max.text(max_time);
    slider_time.attr("max", max_time);
    slider_time.val(curr_time);
    slider_time.on("input", function() {
        let val = $('#slider-time').val();
        Volrend.set_time(parseInt(val));
         $('#slider-time-label-curr').text(val)
    });
};

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

    $('#layers-close').click(function() {
        $('#layers').css('display', 'none');
    });

    $('#layers-btn').click(function() {
        // populateLayers();
        $('#layers').css('display', 'block');
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
    sliders_grid_reso.val(4);
    sliders_grid_reso.on("input", function() {
        let value = parseInt(document.getElementById('slider-grid-reso').value);
        let opt = Volrend.get_options();
        opt.grid_max_depth = value;
        Volrend.set_options(opt);
        document.getElementById('grid-reso-disp').innerText = value;
    });

    $('#mesh-add-cube-btn').click(function() {
        // Position, scale, color
        Volrend.mesh_add_cube([0.0, 0.0, 1.0], 0.2, [1.0, 0.5, 0.2]);
        populateLayers();
    });
    $('#mesh-add-sphere-btn').click(function() {
        Volrend.mesh_add_sphere([0.4, 0.0, 1.0], 0.1, [1.0, 0.0, 0.0]);
        populateLayers();
    });
};

let guiLoadTreeUpdate = function() {
    let sliders_decomp = $('.slider-decomp');
    let max_basis = Math.max(Volrend.get_basis_dim() - 1, 0);
    sliders_decomp.attr('max', max_basis);
    $('#slider-decomp-min').val(0);
    $('#slider-decomp-max').val(max_basis);
    document.getElementById('decomp-min-disp').innerText = '0';
    document.getElementById('decomp-max-disp').innerText = max_basis;
};
