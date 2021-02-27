var Util = {
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
    icon_caret_down: '<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7,10L12,15L17,10H7Z" /></svg>',
    icon_caret_up: '<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M7,15L12,10L17,15H7Z" /></svg>',
    icon_play: '<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z" /></svg>',
    icon_stop: '<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M18,18H6V6H18V18Z" /></svg>',
    icon_delete: '<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z" /></svg>',
};
