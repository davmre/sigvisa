*SIG-VISA*
SIGnal-based Vertically Integrated Seismic Analysis

* INSTALLATION *

See INSTALL for setup instructions.

* WEB INTERFACE *

To start the web interface, cd to web/ and run

python manage.py runserver_plus

*USAGE EXAMPLES*

Say I want to be able to locate a particular evid.
First, find stations detecting this event:
python explore/clearest_detections_of_evid.py --evid=5270227 --ss_only

Create a fitting run evid list with stations you choose:
python learn/filter_evids_for_run.py -s FITZ,KAPI,STKA,NWAO,CTA -o evid_list --min_mb=4.5 --start_time=1072936800 --end_time=1237680000 --only_phases=P

Do the fitting run:
utils/bgjob python learn/batch_fit_from_evids.py -e evid_list --run_name pipeline --run_iter=1 --fit_wiggles --chan=BHZ --nm_type=ar

Train GP coda models:
python learn/train_coda_models.py -s FITZ,KAPI,STKA,NWAO,CTA -r pipeline -p P -t coda_decay,amp_transfer,peak_offset

Evaluate GP code models:
python learn/cross_validation.py -s CTA -r pipeline -i 1 -p P
and so on for other stations...

Extract wiggles:
python learn/extract_wiggles.py -r pipeline -i 1


Locate event:
python infer/gridsearch.py -e 5270227 -s FITZ,KAPI,STKA,NWAO,CTA -r pipeline --phases=P --method=mode


*FILES*

__init__.py -- singleton class SignalModel keeps a C sigmodel object, database connection, cached waveforms, whatever else we need.
    learn/
        batch_fit_from_evids.py -- takes a file written by filter_evids_for_run, and finds a template fit for each (sta, evid) pair, in parallel.
	cross_validation.py
	cross_validation_aggregate_results.py
        filter_evids_for_run.py -- writes a list of (sta, evid) pairs filtered to specific time/magnitude/phase ranges.
        train_coda_models.py --  train param models from template fits
        extract_wiggles.py
        train_wiggles.py
        fit_shape_params.py -- finds template fits for a station, in an abstract way (with specific hinting moved to the template model)
    infer/
        optimize/
        optimize_util.py
        gridsearch.py --  takes an event, a set of stations, does a grid search and outputs a location and plotted heatmaps (DONE)
    models/
        envelope_model.py -- given a TemplateModel and the C code for computing signal likelihood, integrate (or optimize, etc) over template params to get prob(envelopes|events) (DONE)
        spatial_regression/
            spatial_regression_model.py
            baseline_models.py
            SpatialGP.py
        noise/
            noise_models.py
            armodel/
        templates/
            template_model.py -- define an abstract TemplateModel class
            paired_exp.py -- define params, fitting, cost, and generation for an exponential template model.
            …
        wiggles/
            wiggle.py -- extract wiggles from templates / create wiggled templates
            wiggle_models.py
            featurizer.py
            fourier_features.py
    plotting/
        heatmap.py -- implements HeatMap class which calls some function over a range of spatial locations and plots the output. (DONE)
        event_heatmap.py -- HeatMap specialized for plotting event location densities, with code to plot stations, other events, the true event locatation, and compute distance between true event location and the most likely location. (DONE)
        plot.py -- basic methods for plotting waveforms and segments (with labeled phase detections) (DONE)
    signals/
        io.py -- code for loading signals (DONE)
        common.py -- definitions of Waveform and Segment, and code for filtering waveforms. (DONE except for masking edge effects of bandpass, and probably other masking weirdness)
        mask_util.py
    source/
        event.py -- defines an event object, including basic properties (location, depth, magnitude, natural or explosion) (DONE)
        brune_source.py -- Brunce source spectrum model for natural events (DONE)
        mm_source.py -- Mueller-Murphy source spectrum model for explosions (DONE)
    tests/
	...
