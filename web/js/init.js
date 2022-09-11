var Volrend = {
    preRun: [],
    postRun: [],
    print: (function() {})(),
    printErr: function() {},
    canvas: (function() {
        var canvas = document.getElementById('canvas');
        canvas.addEventListener("webglcontextlost", function(e) {
            e.preventDefault(); }, false);
        return canvas;
    })(),
    setStatus: function() {},
    totalDependencies: 0,
    monitorRunDependencies: function() {},
    onRuntimeInitialized: function() { $(document).ready(() => {
		let onResizeCanvas = function() {
			let canvas = document.getElementById("canvas");
			let height = window.innerHeight;
			let width = window.innerWidth;
			canvas.width = width;
			canvas.height = height;
			Volrend.on_resize(width, height);
		};
		// Resize handler
		window.addEventListener('resize', onResizeCanvas, false);

		document.addEventListener('keydown', function(e){
			// Must either have ctrl down OR nothing selected to activate
			if (event.ctrlKey===false &&
				$('*:focus').length > 0) return;
			if (event.ctrlKey===true &&
				(event.which === 61 || event.which === 107 ||
					event.which === 173 || event.which === 109  || event.which === 187  ||
					event.which === 189 || event.which === 72 ||
					event.which === 79 || event.which === 80)) {
				event.preventDefault();
			}
			Volrend.on_key(e.which, e.ctrlKey, e.shiftKey, e.altKey);
		});
		document.addEventListener("mousedown", function(e){
			Volrend.on_mousedown(e.clientX, e.clientY, (e.button === 1) || (e.button === 2));
		});
		var pinch_zoom_dist = -1.;
		var touch_cen_x = -1.;
		var touch_cen_y = -1.;
		document.addEventListener("touchstart", function(e){
			if (e.touches.length > 0) {
				touch_cen_x = 0.; touch_cen_y = 0.;
				for (let i = 0; i < e.touches.length; i++) {
					touch_cen_x += e.touches[i].pageX;
					touch_cen_y += e.touches[i].pageY;
				}
				touch_cen_x /= e.touches.length;
				touch_cen_y /= e.touches.length;
				if (e.touches.length === 2) {
					let touch1 = e.touches[0],
						touch2 = e.touches[1];
					let dist = Math.hypot(
						touch1.pageX - touch2.pageX,
						touch1.pageY - touch2.pageY);
					pinch_zoom_dist = dist;
				}
				Volrend.on_mousedown(touch_cen_x, touch_cen_y, e.touches.length > 1);
			}
		});
		document.addEventListener("mousemove", function(e){
			if (Volrend.is_camera_moving()) {
				Volrend.on_mousemove(e.clientX, e.clientY);
			}
		});
		document.addEventListener("touchmove", function(e){
			if (e.touches.length > 0) {
				touch_cen_x = 0.; touch_cen_y = 0.;
				for (let i = 0; i < e.touches.length; i++) {
					touch_cen_x += e.touches[i].pageX;
					touch_cen_y += e.touches[i].pageY;
				}
				touch_cen_x /= e.touches.length;
				touch_cen_y /= e.touches.length;
				if (e.touches.length === 2) {
					let touch1 = e.touches[0],
						touch2 = e.touches[1];
					let dist = Math.hypot(
						touch1.pageX - touch2.pageX,
						touch1.pageY - touch2.pageY);
					if (pinch_zoom_dist > 0.001) {
						let delta_y = pinch_zoom_dist - dist;
						Volrend.on_mousewheel(delta_y < 0,
							Math.max(Math.abs(delta_y) * 0.05, 0.0001),
							touch_cen_x, touch_cen_y);
					}
					pinch_zoom_dist = dist;
				}
				Volrend.on_mousemove(touch_cen_x, touch_cen_y);
			}
		});
		document.addEventListener("mouseup", function(){ Volrend.on_mouseup(); });
		document.addEventListener("touchend", function(){
			Volrend.on_mouseup();
			pinch_zoom_dist = -1.;
		});
		document.addEventListener("wheel", function(e){
			Volrend.on_mousewheel(e.deltaY < 0,
				1.0, e.offsetX, e.offsetY);
			event.preventDefault();
		});

		// Emscripten GLFW monkey patch
		// - removed undesired fullscreen request on resize
		// - remove e.preventDefault() in keydown handler which broke backspace
		(function() {
			GLFW.setWindowSize = function(winid, width, height) {
				var win = GLFW.WindowFromId(winid);
				if (!win) return;
				// Removed requestFUllscreen, which caused error/weird behavior
				if (GLFW.active.id == win.id) {
					Browser.setCanvasSize(width, height);
					win.width = width;
					win.height = height
				}
				if (!win.windowSizeFunc)
					return;
				dynCall_viii(win.windowSizeFunc, win.id, width, height)
			};
			window.removeEventListener("keydown", GLFW.onKeydown, true);
			GLFW.onKeydown = function (event) {
				// No PreventDefault on backspace
				GLFW.onKeyChanged(event.keyCode, 1);
			};
			window.addEventListener("keydown", GLFW.onKeydown, true);
		})();
        onResizeCanvas();

        $('.load-remote-scene').click(function() {
            let remote_path = this.getAttribute("data");
            Volrend.load_remote(remote_path);
        });

        let hide_layers = Volrend.util.findGetParameter('hide_layers');
        if (hide_layers !== null && hide_layers === "1") {
            Volrend.layer_set_default_visible(false);
        }

        let init_load_paths = Volrend.util.findGetParameter('load');
        if (init_load_paths !== null) {
            init_load_paths = init_load_paths.split(';');
            for (var i = 0; i < init_load_paths.length; i++) {
                Volrend.load_remote(init_load_paths[i]);
            }
        }

        const event = new Event('volrend_ready');
        window.dispatchEvent(event);
    }); },
    set_title: function(title) {
        $('#nerfvis-title').text(title);
        document.title = title;
    },
    _reportProgress: function(x) {
        let prog_ele = $("#load-text");
        prog_ele.text(x.toFixed(2)) + "%";
    },
    _reportComplete: function() {
        let loading_ele = $('#loading');
        setTimeout(function() {
            loading_ele.css('opacity', '0');
            setTimeout(function() {
                loading_ele.css('display', 'none');
            }, 600);
        }, 50);
        Volrend.layer_set_visible(i, false);
        Volrend._populateLayers();
    },
    _reportError: function() {
        let prog_ele = $("#load-text");
        prog_ele.text("ERROR");
    },
    _showLoadingScreen: function() {
        let loading_ele = $('#loading');
        loading_ele.css('display', 'block');
        setTimeout(function() {
            loading_ele.css('opacity', '1');
        }, 10);
    },
	_populateLayers: function() {
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
		const volume_class = "layers-item-volume";
		const mesh_class = "layers-item-mesh";

		let layer_cnt = Volrend.layer_count();
		layer_tree = {__leaf : false};
		for (let i = 0; i < layer_cnt; i++) {
			let layer_name = Volrend.layer_get_name(i);
			if (layer_name.length == 0 || layer_name[0] == '_') {
				// Hide from legend
				continue;
			}
			let layer_color = Volrend.layer_get_color(i);
			let layer_is_visible = Volrend.layer_get_visible(i);
			let layer_color_str = "rgba(" + layer_color[0] * 255.0 + ","
										 + layer_color[1] * 255.0 + ","
										 + layer_color[2] * 255.0 + ")";
			let layer_bg_color_str = layer_is_visible ? layer_color_str : "#fff";
			let classes_str = " " + (Volrend.layer_is_volume(i) ? volume_class : mesh_class);
			classes_str += layer_is_visible ? "" : " " + invis_class;

			let layer_name_tree = layer_name.split('/');

			let it = layer_tree;

			for (let i = 0; i < layer_name_tree.length - 1; i++) {
				let name_part = layer_name_tree[i];
				if (!(name_part in it)) {
					it[name_part] = {
						__leaf : false,
						__name : name_part
					};
				}
				it = it[name_part];
			}
			let last_name = layer_name_tree[layer_name_tree.length-1];
			it[last_name] = {
				__leaf : true,
				__name : last_name,
				__full_name : layer_name,
				color : layer_color,
				is_visible : layer_is_visible,
				color_str : layer_color_str,
				bg_color_str : layer_bg_color_str,
				classes_str : classes_str,
				layer_id : i
			};

		}

		let dfs_populate_html = function(obj, full_name, depth) {
			if (obj.__leaf) {
				html += template.replace(re_name, obj.__name)
					.replace(re_full_name, obj.full_name)
					.replace(re_id, obj.layer_id)
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
				Volrend.layer_set_visible(obj.layer_id, new_visible);
				let base_ele = $("#layer-item-" + obj.layer_id);
				let color_ele = $("#layer-item-color-" + obj.layer_id);
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

		layers_list.html(html);
		$('.layers-item-show-grid').click(function(e) {
			let $this = $(this);
			let layer_id = parseInt($this.parent().attr("layer"));
			let new_show_grid = !Volrend.layer_get_show_grid(layer_id);
			Volrend.layer_set_show_grid(layer_id, new_show_grid);
			if (new_show_grid) {
				$this.addClass("layers-item-show-grid-visible");
			} else {
				$this.removeClass("layers-item-show-grid-visible");
			}
			e.stopPropagation();
		});

		$('.layers-item').click(function() {
			let $this = $(this);
			let color_ele = $this.children('.layers-item-color');
			let layer_id = parseInt($this.attr("layer"));
			let new_visible = !Volrend.layer_get_visible(layer_id);
			Volrend.layer_set_visible(layer_id, new_visible);
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
		let max_time = Volrend.layer_max_time();
		slider_time_label_curr.text(curr_time);
		slider_time_label_max.text(max_time);
		slider_time.attr("max", max_time);
		slider_time.val(curr_time);
		slider_time.on("input", function() {
			let val = $('#slider-time').val();
			Volrend.set_time(parseInt(val));
			 $('#slider-time-label-curr').text(val)
		});
	},
    util: {
        download : function(filename, text) {
            var element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
            element.setAttribute('download', filename);

            element.style.display = 'none'; document.body.appendChild(element);
            element.click();

            document.body.removeChild(element);
        },
        is_mobile:
        ( /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)),
        getFile: function(url, callback) {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, true);
            xhr.responseType = 'text';
            xhr.onload = function() {
                var status = xhr.status;
                if (status === 200) {
                    callback(null, xhr.response);
                } else {
                    callback(status, xhr.response);
                }
            };
            xhr.send();
        },
        isObject: function(value) {
            var type = typeof value;
            return value != null && (type == 'object' || type == 'function');
        },
        debounce: function(func, wait, options) {
            var lastArgs,
                lastThis,
                maxWait,
                result,
                timerId,
                lastCallTime,
                lastInvokeTime = 0,
                leading = false,
                maxing = false,
                trailing = true;

            if (typeof func != 'function') {
                throw new TypeError(FUNC_ERROR_TEXT);
            }
            wait = parseInt(wait) || 0;
            if (Volrend.util.isObject(options)) {
                leading = !!options.leading;
                maxing = 'maxWait' in options;
                maxWait = maxing ? Math.max(parseInt(options.maxWait) || 0, wait) : maxWait;
                trailing = 'trailing' in options ? !!options.trailing : trailing;
            }

            function invokeFunc(time) {
                var args = lastArgs,
                    thisArg = lastThis;

                lastArgs = lastThis = undefined;
                lastInvokeTime = time;
                result = func.apply(thisArg, args);
                return result;
            }

            function leadingEdge(time) {
                // Reset any `maxWait` timer.
                lastInvokeTime = time;
                // Start the timer for the trailing edge.
                timerId = setTimeout(timerExpired, wait);
                // Invoke the leading edge.
                return leading ? invokeFunc(time) : result;
            }

            function remainingWait(time) {
                var timeSinceLastCall = time - lastCallTime,
                    timeSinceLastInvoke = time - lastInvokeTime,
                    timeWaiting = wait - timeSinceLastCall;

                return maxing
                    ? Math.min(timeWaiting, maxWait - timeSinceLastInvoke)
                    : timeWaiting;
            }

            function shouldInvoke(time) {
                var timeSinceLastCall = time - lastCallTime,
                    timeSinceLastInvoke = time - lastInvokeTime;

                // Either this is the first call, activity has stopped and we're at the
                // trailing edge, the system time has gone backwards and we're treating
                // it as the trailing edge, or we've hit the `maxWait` limit.
                return (lastCallTime === undefined || (timeSinceLastCall >= wait) ||
                    (timeSinceLastCall < 0) || (maxing && timeSinceLastInvoke >= maxWait));
            }

            function timerExpired() {
                var time = Date.now();
                if (shouldInvoke(time)) {
                    return trailingEdge(time);
                }
                // Restart the timer.
                timerId = setTimeout(timerExpired, remainingWait(time));
            }

            function trailingEdge(time) {
                timerId = undefined;

                // Only invoke if we have `lastArgs` which means `func` has been
                // debounced at least once.
                if (trailing && lastArgs) {
                    return invokeFunc(time);
                }
                lastArgs = lastThis = undefined;
                return result;
            }

            function cancel() {
                if (timerId !== undefined) {
                    clearTimeout(timerId);
                }
                lastInvokeTime = 0;
                lastArgs = lastCallTime = lastThis = timerId = undefined;
            }

            function flush() {
                return timerId === undefined ? result : trailingEdge(Date.now());
            }

            function debounced() {
                var time = Date.now(),
                    isInvoking = shouldInvoke(time);

                lastArgs = arguments;
                lastThis = this;
                lastCallTime = time;

                if (isInvoking) {
                    if (timerId === undefined) {
                        return leadingEdge(lastCallTime);
                    }
                    if (maxing) {
                        // Handle invocations in a tight loop.
                        clearTimeout(timerId);
                        timerId = setTimeout(timerExpired, wait);
                        return invokeFunc(lastCallTime);
                    }
                }
                if (timerId === undefined) {
                    timerId = setTimeout(timerExpired, wait);
                }
                return result;
            }
            debounced.cancel = cancel;
            debounced.flush = flush;
            return debounced;
        },
        findGetParameter : function(parameterName) {
            var result = null,
                tmp = [];
            location.search
                .substr(1)
                .split("&")
                .forEach(function (item) {
                    tmp = item.split("=");
                    if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
                });
            return result;
        },
    },
    // DO NOT CHANGE THE FOLLOWING LINE, inline.py will add the wasm there
    wasmBinary: Uint8Array.from(atob("{{inline_wasm_b64}}"), c => c.charCodeAt(0)).buffer,
};

