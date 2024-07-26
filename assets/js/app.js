let slideIndex = 1;
showSlides(slideIndex);
console.log(window.matchMedia("(min-width: 64em)").matches);
console.log(window.matchMedia("(min-width: 48em)").matches);
// Next/previous controls
function plusSlides(n) {
  showSlides(slideIndex += n);
}

// Thumbnail image controls
function currentSlide(n) {
  showSlides(slideIndex = n);
}
window.onscroll = function() {
        	scrollFunction()
        };
function showSlides(n) {
  let i;
  let slides = document.getElementsByClassName("mySlides");
  let dots = document.getElementsByClassName("dot");
  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " active";
}

const max_scroll_position = 180;


function scrollFunction() {
    if (document.body.scrollTop > max_scroll_position || document.documentElement.scrollTop > max_scroll_position) {
       collapse();
    }
    else {
        expand()
    }
  }



  function collapse() {
    document.getElementById("authors").style.display = "none";
    // Project name
    document.getElementsByClassName("project-name")[0].style.fontSize = "1.3rem";
    document.getElementsByClassName("project-name")[0].style.position = "absolute";
    document.getElementsByClassName("project-name")[0].style.display = "inline-block";
    document.getElementsByClassName("project-name")[0].style.top = "25px";
    document.getElementsByClassName("project-name")[0].style.left = "10px";
    document.getElementsByClassName("project-name")[0].style.lineHeight = "1";
    document.getElementsByClassName("project-name")[0].style.textAlign = "left";


     if (window.matchMedia("(min-width: 64em)").matches) {

        document.getElementsByClassName("project-name")[0].style.fontSize = "1.3rem";
        document.getElementsByTagName("header")[0].style.Height = "5rem";

    document.getElementsByClassName("project-name")[0].style.marginRight = "150px";
        document.getElementsByTagName("header")[0].style.maxHeight = "5rem";
        let linkImages = document.getElementById("links").getElementsByTagName("img")
    for (let i = 0; i < linkImages.length; i++) {
        linkImages[i].style.height = "40px";
    }
    } else if (window.matchMedia("(min-width: 48em)").matches) {
        document.getElementsByClassName("project-name")[0].style.fontSize = "1.15rem";

    document.getElementsByClassName("project-name")[0].style.marginRight = "150px";
        document.getElementsByTagName("header")[0].style.Height = "6.5rem";
        document.getElementsByTagName("header")[0].style.maxHeight = "6.5rem";
        let linkImages = document.getElementById("links").getElementsByTagName("img")
    for (let i = 0; i < linkImages.length; i++) {
        linkImages[i].style.height = "40px";
    }
    } else {
        document.getElementsByClassName("project-name")[0].style.fontSize = "0.65rem";
        let linkImages = document.getElementById("links").getElementsByTagName("img")
    for (let i = 0; i < linkImages.length; i++) {
        linkImages[i].style.height = "25px";
        linkImages[i].style.marginTop = "10px";
    document.getElementsByClassName("project-name")[0].style.marginRight = "80px";
    }
    }



    document.getElementById("small-authors").style.display = "none";


    // Header
    document.getElementsByTagName("header")[0].style.maxHeight = "73px";
    document.getElementsByTagName("header")[0].style.Height = "73px";

    // Shrinking links

    let linkLabels = document.getElementById("links").getElementsByTagName("h4")
    for (let i = 0; i < linkLabels.length; i++) {
        linkLabels[i].style.display = "none";
    }
    // Changing position of links container
    document.getElementById("links").style.position = "absolute";
    document.getElementById("links").style.top = "5px";
    document.getElementById("links").style.right = "0";

    // Remove tagline
    document.getElementById("project-tagline").style.display = "none";

    // Collapsed authors list
      document.getElementById("collapsed-authors").style.display = "inline";
      document.getElementById("collapsed-authors").style.opacity = "1";
      document.getElementById("collapsed-authors").style.visibility = "visible";

  }

  function expand(){
    document.getElementById("links").style.display = "flex";

    document.getElementsByClassName("project-name")[0].style.float = "none";
    document.getElementsByClassName("project-name")[0].style.display = "block";
    document.getElementsByClassName("project-name")[0].style.position = "static";
    document.getElementsByClassName("project-name")[0].style.marginRight = "0px";
    document.getElementsByClassName("project-name")[0].style.lineHeight = "1.5";

    document.getElementsByClassName("project-name")[0].style.textAlign = "center";
    if (window.matchMedia("(min-width: 64em)").matches) {
        document.getElementsByClassName("project-name")[0].style.fontSize = "3.25rem";
        document.getElementsByTagName("header")[0].style.Height = "28rem";
    document.getElementsByTagName("header")[0].style.maxHeight = "28rem";

    document.getElementById("authors").style.display = "flex";
    let linkImages = document.getElementById("links").getElementsByTagName("img")
    for (let i = 0; i < linkImages.length; i++) {
        linkImages[i].style.height = "60px";
    }
    } else if (window.matchMedia("(min-width: 48em)").matches) {
        document.getElementsByClassName("project-name")[0].style.fontSize = "2.25rem";
        document.getElementsByTagName("header")[0].style.Height = "30rem";
    document.getElementsByTagName("header")[0].style.maxHeight = "30rem";

    document.getElementById("authors").style.display = "flex";
    let linkImages = document.getElementById("links").getElementsByTagName("img")
    for (let i = 0; i < linkImages.length; i++) {
        linkImages[i].style.height = "60px";
    }
    } else {
        document.getElementsByClassName("project-name")[0].style.fontSize = "1.33rem";
        document.getElementsByTagName("header")[0].style.Height = "23rem";

    document.getElementsByTagName("header")[0].style.maxHeight = "23rem";
    document.getElementById("small-authors").style.display = "block";

    let linkImages = document.getElementById("links").getElementsByTagName("img")
    for (let i = 0; i < linkImages.length; i++) {
        linkImages[i].style.height = "40px";
    }


    }

    let linkLabels = document.getElementById("links").getElementsByTagName("h4")
    for (let i = 0; i < linkLabels.length; i++) {
        linkLabels[i].style.display = "block";
    }

    document.getElementById("project-tagline").style.display = "block";
    document.getElementById("links").style.position = "static";

    document.getElementById("collapsed-authors").style.display  = "none";
      document.getElementById("collapsed-authors").style.opacity = "0";
      document.getElementById("collapsed-authors").style.visibility = "hidden";
  }
