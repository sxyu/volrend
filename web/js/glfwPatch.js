// Emscripten GLFW monkey patch
// - removed undesired fullscreen request on resize
// - remove e.preventDefault() in keydown handler which broke backspace
var glfwPatch = function() {
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
};
