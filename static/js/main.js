(function ($) {
    "use strict";

    // Spinner
    
    var spinner = function () {
        setTimeout(function () {
            if ($('#spinner').length > 0) {
                $('#spinner').removeClass('show');
            }
            else {
                console.error("Spinner element not found!");
            }
        }, 1);
    };
    spinner();



    // Fixed Navbar bg-dark shadow
    $(window).scroll(function () {
        if ($(window).width() < 992) {
            if ($(this).scrollTop() > 45) {
                $('.fixed-top').addClass('custom-bg shadow');
            } else {
                $('.fixed-top').removeClass('custom-bg shadow');
            }
        } else {
            if ($(this).scrollTop() > 45) {
                $('.fixed-top').addClass('custom-bg shadow').css('top', -45);
            } else {
                $('.fixed-top').removeClass('custom-bg shadow').css('top', 0);
            }
        }
    });


})(jQuery);

