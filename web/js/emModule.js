var cppReportProgress = function(x) {
    let prog_ele = $("#load-progress");
    prog_ele.css("width", x + "%");
    prog_ele.attr("aria-valuenow", x);
    prog_ele.text(Math.round(x * 100) / 100);
    if (x > 100.0) {
        let loading_ele = $('#loading');
        setTimeout(function() {
            loading_ele.css('opacity', '0');
            setTimeout(function() {
                loading_ele.css('display', 'none');
            }, 1000);
        }, 100);
    }
};

var onResizeCanvas = function() {
    let canvas = document.getElementById("canvas");
    let height = window.innerHeight - $('#header').outerHeight() - 7;
    let width = window.innerWidth;
    canvas.width = width;
    canvas.height = height;
    Volrend.on_resize(width, height);
    Volrend.delayedRedraw();
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
};
