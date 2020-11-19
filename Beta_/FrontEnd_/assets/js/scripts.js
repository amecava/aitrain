/* ---------------------------------------------
 common scripts
 --------------------------------------------- */

;(function () {

    "use strict"; // use strict to start

    $(document).ready(function () {


        /* ---------------------------------------------
         Configure tooltips for collapsed side navigation
         --------------------------------------------- */

        $('.left-side-nav [data-toggle="tooltip"]').tooltip({
            template: '<div class="tooltip left-side-nav-tooltip" role="tooltip"><div class="arrow"></div><div class="tooltip-inner"></div></div>'
        });

        /* ---------------------------------------------
         Toggle the side navigation
         --------------------------------------------- */

        $("#left-nav-toggler").on('click',function(e){
            e.preventDefault();
            $("body").toggleClass("left-side-toggled");
            $(".left-side-nav .nav-link-collapse").addClass("collapsed");
            $(".left-side-nav .sidenav-second-level, .left-side-nav .sidenav-third-level").removeClass("show");
        });

        $("#sidenavToggler").on('click',function(e){
            e.preventDefault();
            $("body").toggleClass("sidenav-toggled");
            $(".navbar-sidenav .nav-link-collapse").addClass("collapsed");
            $(".navbar-sidenav .sidenav-second-level, .navbar-sidenav .sidenav-third-level").removeClass("show");
        });

        /* ---------------------------------------------
         Force the toggled class to be removed when a
         collapsible nav link is clicked
         --------------------------------------------- */

        $(".left-side-nav .nav-link-collapse").on('click',function(e){
            e.preventDefault();
            $("body").removeClass("left-side-toggled");
        });


        /* ---------------------------------------------
         Prevent the content wrapper from scrolling
         when the fixed side navigation hovered over
         --------------------------------------------- */

        $('body.fixed-nav .left-side-nav, body.fixed-nav .sidenav-toggler, body.fixed-nav .navbar-collapse').on('mousewheel DOMMouseScroll', function(e) {
            var e0 = e.originalEvent,
                delta = e0.wheelDelta || -e0.detail;
            this.scrollTop += (delta < 0 ? 1 : -1) * 30;
            e.preventDefault();
        });


        $(".right_side_toggle").on('click', function(){
            $('#right_side_bar').toggleClass('show');
        });


        /* ---------------------------------------------
         Accordion init
         --------------------------------------------- */

        var allPanels = $(".accordion > dd").hide();
        allPanels.first().slideDown("easeOutExpo");
        $(".accordion").each(function () {
            $(this).find("dt > a").first().addClass("active").parent().next().css({display: "block"});
        });

        $(".accordion > dt > a").click(function () {

            var current = $(this).parent().next("dd");
            $(this).parents(".accordion").find("dt > a").removeClass("active");
            $(this).addClass("active");
            $(this).parents(".accordion").find("dd").slideUp("easeInExpo");
            $(this).parent().next().slideDown("easeOutExpo");

            return false;

        });


        /* ---------------------------------------------
         Toggle init
         --------------------------------------------- */

        var allToggles = $(".toggle > dd").hide();
        $(".toggle > dt > a").click(function () {

            if ($(this).hasClass("active")) {

                $(this).parent().next().slideUp("easeOutExpo");
                $(this).removeClass("active");

            }
            else {
                var current = $(this).parent().next("dd");
                $(this).addClass("active");
                $(this).parent().next().slideDown("easeOutExpo");
            }

            return false;
        });

        /* ---------------------------------------------
         Configure tooltips globally
         --------------------------------------------- */

        $('[data-toggle="tooltip"]').tooltip()


        /* ---------------------------------------------
         Configure popover globally
         --------------------------------------------- */
        $('[data-toggle="popover"]').popover();


        /* ---------------------------------------------
         Custom scroll configuration
         --------------------------------------------- */

        $(".chat-wrap").mCustomScrollbar({
            autoHideScrollbar: true,
            scrollInertia: 0
        });



    });

})(jQuery);

function onSuccess(googleUser) {
    var profile = googleUser.getBasicProfile();
    gapi.client.load('plus', 'v1', function () {
        var request = gapi.client.plus.people.get({
            'userId': 'me'
        });
        //Display the user details
        request.execute(function (resp) {
            var profileHTML = '<div class="profile"><div class="head">Welcome '+resp.name.givenName+'! <a href="javascript:void(0);" onclick="signOut();">Sign out</a></div>';
            profileHTML += '<img src="'+resp.image.url+'"/><div class="proDetails"><p>'+resp.displayName+'</p><p>'+resp.emails[0].value+'</p><p>'+resp.gender+'</p><p>'+resp.id+'</p><p><a href="'+resp.url+'">View Google+ Profile</a></p></div></div>';
            $('.userContent').html(profileHTML);
            $('#gSignIn').slideUp('slow');
        });
    });
}
function onFailure(error) {
    alert(error);
}
function renderButton() {
    gapi.signin2.render('gSignIn', {
        'scope': 'profile email',
        'width': 240,
        'height': 50,
        'longtitle': true,
        'theme': 'dark',
        'onsuccess': onSuccess,
        'onfailure': onFailure
    });
}
function signOut() {
    var auth2 = gapi.auth2.getAuthInstance();
    auth2.signOut().then(function () {
        $('.userContent').html('');
        $('#gSignIn').slideDown('slow');
    });
}
