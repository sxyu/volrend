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
		if (Util.isObject(options)) {
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
};
