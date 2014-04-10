__author__ = "Nuno Lages"
__email__ = "lages@uthscsa.edu"


import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import scipy as sp
import scipy.stats as stats
import scipy.io as sio
from time import time


class TruncThresholdObjects(cpm.CPModule):

    variable_revision_number = 1
    module_name = "TruncThresholdObjects"
    category = "Image Processing"

    def create_settings(self):

        self.input_image_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Input image name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>ImageTemplate</b> will do something to this image.
            """
        )

        self.output_image_name = cps.ImageNameProvider(
            "Output image name:",
            # The second parameter holds a suggested name for the image.
            "OutputImage",
            doc="""This is the image resulting from the operation."""
        )

        self.input_objects_name = cps.ObjectNameSubscriber(
            # The text to the left of the edit box
            "Input objects name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the objects that the module operates on. You can
            choose any objects that is made available by a prior module.
            <br>
            <b>TruncThresholdObjects</b> will do something to this objects.
            """
        )

        self.center = cps.Choice(
            "Center choice:",
            # The choice takes a list of possibilities. The first one
            # is the default - the one the user will typically choose.
            ['median', 'average'],
            doc="""Choose what to use as estimate of the mean of the
                truncated normal distribution."""
        )

        self.scale_r = cps.Float(
            "Truncated normal parameter red channel:",
            # The default value
            0.05,
            doc=""""""
        )

        self.scale_g = cps.Float(
            "Truncated normal parameter green channel:",
            # The default value
            0.05,
            doc=""""""
        )

        self.scale_b = cps.Float(
            "Truncated normal parameter blue channel:",
            # The default value
            0.05,
            doc=""""""
        )

        self.percentile_r = cps.Float(
            "Percentile red channel:",
            # The default value
            0.01,
            doc=""""""
        )

        self.percentile_g = cps.Float(
            "Percentile green channel:",
            # The default value
            0.01,
            doc=""""""
        )

        self.percentile_b = cps.Float(
            "Percentile blue channel:",
            # The default value
            0.0,
            doc=""""""
        )

    def settings(self):
        return [self.input_image_name,
                self.output_image_name,
                self.input_objects_name,
                self.center,
                self.scale_r,
                self.scale_g,
                self.scale_b,
                self.percentile_r,
                self.percentile_g,
                self.percentile_b]

    def run(self, workspace):

        t0 = time()

        diagnostics = dict()

        cent = self.center.get_value()

        input_objects_name = self.input_objects_name.value
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)

        input_image_name = self.input_image_name.value
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        output_image_name = self.output_image_name.value

        input_image = image_set.get_image(input_image_name)# must_be_rgb=True)
        pixels = input_image.pixel_data
        diagnostics['pixels'] = pixels

        input_objects = object_set.get_objects(input_objects_name)

        mask = input_objects.get_segmented()

        new_im = sp.zeros(shape=pixels.shape)

        diagnostics['new_im'] = list()
        diagnostics['nucleus_processed'] = list()
        diagnostics['nucleus_pixels'] = list()
        diagnostics['ci'] = list()
        diagnostics['mu'] = list()
        diagnostics['sigma'] = list()
        diagnostics['sigma2'] = list()
        diagnostics['old_sigma'] = list()
        diagnostics['a'] = list()
        diagnostics['b'] = list()
        diagnostics['x1'] = list()
        diagnostics['x2'] = list()
        diagnostics['cx'] = list()
        diagnostics['yhat'] = list()

        diagnostics['time_first_part'] = time() - t0

        for x in range(1, mask.max()+1):

            t0 = time()

            nucleus_map = mask == x


            nucleus_pixels = \
                sp.multiply(pixels, nucleus_map[:, :, sp.newaxis] > 0)

            diagnostics['times_loop_' + str(x) + '_nditer'] = time() - t0
            t0 = time()

            diagnostics['nucleus_pixels'].append(nucleus_pixels)

            nucleus_pixels_t = sp.transpose(nucleus_pixels)

            # nucleus_ci_r = get_ci(nucleus_pixels_t[0],
            nucleus_ci_r, mu_r, sigma_r, a_r, b_r, old_sigma_r, x1_r, x2_r, \
            cx_r, \
            yhat_r, sigma2_r = get_ci(nucleus_pixels_t[0],
                                  percentile=self.percentile_r.get_value(),
                                  center=cent,
                                  mod=self.scale_r.get_value())

            # nucleus_ci_g = get_ci(nucleus_pixels_t[1],
            nucleus_ci_g, mu_g, sigma_g, a_g, b_g, old_sigma_g, x1_g, x2_g, \
            cx_g, \
            yhat_g, sigma2_g = get_ci(nucleus_pixels_t[1],
                                  percentile=self.percentile_g.get_value(),
                                  center=cent,
                                  mod=self.scale_g.get_value())

            # nucleus_ci_b = get_ci(nucleus_pixels_t[2],
            nucleus_ci_b, mu_b, sigma_b, a_b, b_b, old_sigma_b, x1_b, x2_b, \
            cx_b, \
            yhat_b, sigma2_b = get_ci(nucleus_pixels_t[2],
                                  percentile=self.percentile_b.get_value(),
                                  center=cent,
                                  mod=self.scale_b.get_value())

            diagnostics['times_loop_' + str(x) + '_ci'] = time() - t0
            t0 = time()

            diagnostics['ci'].append((nucleus_ci_r, nucleus_ci_g,
                                      nucleus_ci_b))
            diagnostics['mu'].append((mu_r, mu_g, mu_b))
            diagnostics['sigma'].append((sigma_r, sigma_g, sigma_b))
            diagnostics['sigma2'].append((sigma2_r, sigma2_g, sigma2_b))
            diagnostics['old_sigma'].append(
                (old_sigma_r, old_sigma_g, old_sigma_b))
            diagnostics['a'].append((a_r, a_g, a_b))
            diagnostics['b'].append((b_r, b_g, b_b))
            diagnostics['x1'].append((x1_r, x1_g, x1_b))
            diagnostics['x2'].append((x2_r, x2_g, x2_b))
            diagnostics['cx'].append((cx_r, cx_g, cx_b))
            diagnostics['yhat'].append((yhat_r, yhat_g, yhat_b))

            nucleus_processed = update_image(nucleus_pixels,
                                             nucleus_ci_r,
                                             nucleus_ci_g,
                                             nucleus_ci_b)

            diagnostics['times_loop_' + str(x) + '_update'] = time() - t0

            diagnostics['nucleus_processed'].append(nucleus_processed)

            new_im = new_im + nucleus_processed

            diagnostics['new_im'].append(new_im)

            from os.path import expanduser
            home = expanduser("~")

            sio.savemat(home + '/diagnostics.mat', diagnostics)

        output_image = cpi.Image(new_im, parent_image=input_image)
        image_set.add(output_image_name, output_image)

    def is_interactive(self):
        return False


def var_truncNormal(a, b, mu, sigma, data, mod=3000.0):

    # x1 = (a - mu)/sigma * stats.norm.pdf(a, mu, sigma)
    x1 = (mu - a)/sigma * stats.norm.pdf(a, mu, sigma)
    x2 = (b - mu)/sigma * stats.norm.pdf(b, mu, sigma)

    cx = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)

    yhat = stats.tvar(data, limits=[mu-mod, mu+mod], inclusive=(False, False))
    sigma2 = yhat/((1+(x1-x2)/cx - ((x1-x2)/cx)**2))
    sigma = sp.sqrt(sigma2)

    return sigma, x1, x2, cx, yhat, sigma2


def update_image(original_im, ci_red, ci_green, ci_blue):

    ci_vec = sp.array((ci_red, ci_green, ci_blue))
    ci_matrix = sp.multiply(sp.ones(original_im.shape), ci_vec)
    new_im = sp.multiply(original_im, original_im > ci_matrix)

    return new_im


def get_ci(im_data, center='median', mod=3000.0, percentile=0.01):

    flattened = sp.concatenate(im_data)
    flattened = flattened[sp.nonzero(flattened)]

    if center == 'median':
        mu = sp.median(flattened)
    elif center == 'mean':
        mu = sp.average(flattened)

    old_sigma = stats.tstd(flattened)

    # mod = sp.float_(mod)
    sigma, x1, x2, cx, yhat, sigma2 = var_truncNormal(mu - mod, mu + mod, mu,
                                              old_sigma, flattened, mod=mod)

    ci = 2 * mu - stats.norm.ppf(percentile, mu, sigma)

    return ci, mu, sigma, mu-mod, mu+mod, old_sigma, x1, x2, cx, yhat, sigma2
