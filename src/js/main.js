$(() => {
    $('.styleImagePicker').imagepicker();

    var canvas = document.getElementById('output');
    canvas.width = 500;
    canvas.height = 500;
    var ctx = canvas.getContext('2d');

    $('#transfer').click(() => {

      var contentdownloadurl = $('#contentImageUrl').val();
      var contenturllist = contentdownloadurl.split('/');
      var contentname = contenturllist[contenturllist.length - 1];
      console.log(contentdownloadurl);
      $.ajax({
          url: '/api/download_img',
          method: 'POST',
          contentType: 'application/json',
          data: JSON.stringify({'url' : contentdownloadurl}),
          success: (data) => {
            console.log('successfully downloaded content img: ' + data.img);
          }
      });
      var contenturl = 'instance/' + contentname;

      var styleurl = '';
      if ($('#styleImageUrl').val() == '') {
        styleurl = 'static/imgs/' + $(".styleImagePicker").val();
      } else {
        var styledownloadurl = $('#styleImageUrl').val();
        var styleurllist = styledownloadurl.split('/');
        var stylename = styleurllist[styleurllist.length - 1];
        $.ajax({
            url: '/api/download_img',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({'url' : styledownloadurl}),
            success: (data) => {
              console.log('successfully downloaded style img: ' + data.img);
            }
        });
        styleurl = 'instance/' + stylename;
      }

      var isize = Number($('#contentSize').val());
      var ssize = Number($('#styleSize').val());
      var cw = Number($('#contentWeight').val());
      var sws = $('#styleWeights').val().split(',').map(Number);
      var tw = Number($('#tvWeight').val());

      console.log(contenturl);
      console.log(styleurl);
      console.log(isize);
      console.log(ssize);
      console.log(cw);
      console.log(sws);
      console.log(tw);

      var input = {
        'content_image' : contenturl,
        'style_image' : styleurl,
        'image_size' : isize,
        'style_size' : ssize,
        'content_layer' : 3,
        'content_weight' : cw,
        'style_layers' : [1, 4, 6, 7],
        'style_weights' : sws,
        'tv_weight' : tw
      }

      $('#processing').show();

      $.ajax({
          url: '/api/styletransfer',
          method: 'POST',
          contentType: 'application/json',
          data: JSON.stringify(input),
          success: (data) => {
            // console.log(data.result);
            $('#processing').hide();
            var shape = data.result[0];

            var canvas = document.getElementById('output');
            var ctx = canvas.getContext('2d');

            var imageData = new ImageData(new Uint8ClampedArray(data.result[1]), shape[1], shape[0]);
            ctx.putImageData(imageData, 0, 0);
          }
      });
    });
});
