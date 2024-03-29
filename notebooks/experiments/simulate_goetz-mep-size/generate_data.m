function generate_data(d_output, vec_draw)
assert(nargin > 0, 'Have to provide output directory.');
if nargin == 1, vec_draw = 1:2000;end
rng(0);
if isempty(which('virtualsubjectEIVGenerateSubject'))
    error('Statistical-MEP-Model code has to be placed on the path.');
end
% options
v.str_set = 'set01';  % subfolder name
v.set_additive_noise = nan;
v.set_power = nan;
v.saturation_observed_threshold = nan;  % change to e.g. 0.95
v.n_reps_search = 1e3;
v.n_reps_minmax = 2e4;
v.stimamp_saturation = 1e5;
v.stimamp_intensity = repmat(linspace(0, 1, 64), 8, 1).';
v.n_participant = 16;
v.optim_rng = 0;
v.skip_optimization = true;
v.show_plot = false;

d_base =  fullfile(d_output, v.str_set);

opt_n = 1e5;
options_opt = optimoptions('patternsearch', ...
    'MaxFunctionEvaluations', 10 * opt_n, 'MaxIterations', opt_n, ...
    'UseParallel', true);
options_opt.PollMethod = 'MADSPositiveBasis2N';
options_opt.SearchFcn = 'searchneldermead';
options_opt.InitialMeshSize = 1.0;
options_opt.Display = 'none';
% options_opt = optimset('fminsearch');

d_draws = fullfile(d_base, 'draws');
d_figures = fullfile(d_base, 'figures');
if not(exist(d_draws, 'dir') == 7), mkdir(d_draws); end
if not(exist(d_figures, 'dir') == 7), mkdir(d_figures); end

rng(0);assert((rand - 0.814723686) < 1e-9, "RNG NOT REPLICATING");

stimamp_intensity = v.stimamp_intensity;
n_participant = v.n_participant;

fprintf('Starting...\n');
for ix_draw = vec_draw
    tic;
    str_draw = sprintf('draw%06d', ix_draw);
    parameters = struct;
    rc_mat = zeros(1, n_participant, size(stimamp_intensity, 1), size(stimamp_intensity, 2));
    int_mat = zeros(1, n_participant, size(stimamp_intensity, 1), size(stimamp_intensity, 2));

    for ix_participant = 1:n_participant
        ix_rng = (ix_draw - 1) * n_participant + ix_participant;
        participant = "P" + sprintf('%03d', ix_participant);
        parameter_set_is_invalid = true;
        while_count = 0;
        while parameter_set_is_invalid
            if while_count == 0
                % first iteration uses the well thought out rng
                ix_rng_local = ix_rng;
                rng(ix_rng_local);
            else
                % but future iterations use the well thought out rng as the
                % seed for new rng seeds
                rng(ix_rng);
                vec_rng = randi(randi(intmax), [1, while_count]);
                ix_rng_local = vec_rng(end);
                rng(ix_rng_local);
            end
            local_parameters = virtualsubjectEIVGenerateSubject;
            if isfinite(v.set_power)
                local_parameters(4) = v.set_power;
            end
            if isfinite(v.set_additive_noise)
                local_parameters(7) = v.set_additive_noise;
            end

            % conisder enabling this if you have to draw too many times to
            % get something reasonable.
            %             if isfinite(v.saturation_observed_threshold)
            %                 n_quick = 100;
            %                 val_saturation_quick = mean(virtstimulate(v.stimamp_saturation * ones(n_quick, 1), local_parameters)); % definitely saturated.
            %                 val_at_100_quick = mean(virtstimulate(1 * ones(n_quick, 1), local_parameters)); % get the 100% mean
            %                 if val_at_100_quick <= (val_saturation_quick * v.saturation_observed_threshold)
            %                     while_count = while_count + 1;
            %                     continue;
            %                 end
            %             end

            val_saturation = mean(virtstimulate(v.stimamp_saturation * ones(v.n_reps_minmax, 1), local_parameters)); % definitely saturated.
            val_at_100 = mean(virtstimulate(1 * ones(v.n_reps_minmax, 1), local_parameters)); % get the 100% mean

            if isfinite(v.saturation_observed_threshold)
                if val_at_100 > (val_saturation * v.saturation_observed_threshold)
                    parameter_set_is_invalid = false;
                end
            else
                parameter_set_is_invalid = false;
            end
            while_count = while_count + 1;
        end
        val_baseline = mean(virtstimulate(0 * ones(v.n_reps_minmax, 1), local_parameters)); % definitely 0.
        ix_rng = ix_rng_local;

        % estimate S50
        virtstimulate_norm = @(x, p) (virtstimulate(x, p) - val_baseline)./(val_saturation - val_baseline);
        get_cost = @(x, n_reps, local_parameters) rms(mean(virtstimulate_norm(repmat(x, n_reps, 1), local_parameters)) - 0.5);
        rng(v.optim_rng);
        s50_init_unscaled = 1e-2 * (local_parameters(5) + (local_parameters(3) /(-1 + (local_parameters(2) + 7) / (local_parameters(2) + 7 - log(2) / log(10))))^(1/local_parameters(4)));  % TODO: uncomment this!
        if v.skip_optimization
            s50_unscaled = s50_init_unscaled;
            cost = nan;
        else
            [s50_unscaled, cost] = patternsearch(@(x) get_cost(x, v.n_reps_search, local_parameters), s50_init_unscaled, [], [], [], [], 0, v.stimamp_saturation, [], options_opt);
            % [s50_unscaled, cost] = fminsearch(@(x) get_cost(x, v.n_reps_search, local_parameters), 0.5, options_opt);
        end
        s50  = s50_unscaled * 100;
        s50_init  = s50_init_unscaled * 100;

        parameters(1, ix_participant).id = char(participant);
        parameters(1, ix_participant).s50 = s50;
        parameters(1, ix_participant).val_baseline = val_baseline;
        parameters(1, ix_participant).val_saturation = val_saturation;
        parameters(1, ix_participant).parameters = local_parameters;
        parameters(1, ix_participant).fraction_saturation_at_100 = val_at_100/val_saturation;
        parameters(1, ix_participant).ix_rng = ix_rng;
        parameters(1, ix_participant).ix_draw = ix_draw;
        parameters(1, ix_participant).s50_init = s50_init;
        parameters(1, ix_participant).skip_optimization = v.skip_optimization;
        parameters(1, ix_participant).optimization_converged = cost < 0.01;
        parameters(1, ix_participant).invalid_draw_count = while_count - 1;

        rng(ix_rng);
        rc_samples = virtstimulate(stimamp_intensity, local_parameters);
        rc_individual = rc_samples * 1e3;  % mV
        stimamp_intensity_individual = stimamp_intensity * 1e2;  % percent

        rc_mat(1, ix_participant, :, :) = rc_individual;
        int_mat(1, ix_participant, :, :) = stimamp_intensity_individual;

        % T = table;
        % T.TMSInt = stimamp_intensity(:) * 100;
        % T.muscle_pkpk = rc_samples(:) * 1e3;  % mV
        % T.participant(:) = participant;
        % T.condition(:) = 1;

        if v.show_plot && ix_draw <=10
            figure(1);
            str_figure = sprintf('%s-%s', str_draw, participant);
            clf;
            hold on;
            plot(stimamp_intensity_individual(:), rc_individual(:), 'sk')
            plot(local_parameters(5) * ones(1, 2), get(gca, 'ylim'), 'm')
            box on;
            grid on;
            xlim([0, 200]);
            plot(s50 * ones(1, 2), get(gca, 'ylim'), 'r--')
            plot(get(gca, 'xlim'), val_baseline * ones(1, 2), 'r--')
            plot(get(gca, 'xlim'), val_saturation * ones(1, 2), 'r--')
            title(str_figure)

            drawnow;
            print(fullfile(d_figures, str_figure), '-dpng')
        end
    end
    toc;
    save(fullfile(d_draws, sprintf('%s.mat', str_draw)), 'parameters', 'rc_mat', 'int_mat', 'v');

    %     try
    %         pause(1);
    %         combine_draws(d_base);
    %     catch errt
    %         fprintf('Failed to combine data!\n')
    %     end
end
end