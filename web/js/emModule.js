var showLoadingScreen = function() {
    let loading_ele = $('#loading');
    loading_ele.css('display', 'block');
    setTimeout(function() {
        loading_ele.css('opacity', '1');
    }, 10);
};

var cppReportProgress = function(x) {
    let prog_ele = $("#load-progress");
    prog_ele.css("width", x + "%");
    prog_ele.attr("aria-valuenow", x);
    if (x > 100.0) {
        let loading_ele = $('#loading');
        setTimeout(function() {
            loading_ele.css('opacity', '0');
            setTimeout(function() {
                loading_ele.css('display', 'none');
            }, 600);
        }, 50);
        guiLoadTreeUpdate();
        Volrend.mesh_set_visible(i, false);
        populateLayers();
    } else {
        prog_ele.text(x.toFixed(2));
    }
};

var cppUpdateFPS = function(fps) {
    $('#fps-counter-val').text(fps.toFixed(2));
};

var onResizeCanvas = function() {
    let canvas = document.getElementById("canvas");
    let height = window.innerHeight - $('#header').outerHeight() - 7;
    let width = window.innerWidth;
    canvas.width = width;
    canvas.height = height;
    Volrend.on_resize(width, height);
};

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
    onRuntimeInitialized: function() { $(document).ready(onInit); },
    set_title: function(title) {
        $('#navbar-title').text(title);
        document.title = title + " - PlenOctree Viewer";
    },
};
