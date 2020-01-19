import React, { useEffect, useState } from 'react';
import Axios from 'axios';
import Button from '@material-ui/core/Button';
import {
	CircularProgress,
	Select,
	Snackbar,
	Typography,
	withStyles,
	MenuItem,
	FormControl,
	InputLabel,
} from '@material-ui/core';
import Alert from '@material-ui/lab/Alert';
import MaterialTable from 'material-table';

const styles = {
	predictButton: {
		width: '15rem',
		alignSelf: 'center',
		padding: '1rem',
		borderRadius: '1rem',
		margin: '1rem',
	},
	container: {
		display: 'flex',
		flexDirection: 'column',
		flexGrow: '1',
		justifyContent: 'space-evenly',
		alignItems: 'center',
		padding: '1rem',
	},
	controls: {
		display: 'flex',
		flexDirection: 'row',
		width: '100%',
		alignItems: 'center',
		justifyContent: 'space-evenly',
	},
};

const getColumns = (keys, categoricalValues) =>
	keys
		.map(key =>
			key !== 'tableData'
				? {
						title: key,
						field: key,
						lookup: Object.keys(categoricalValues).includes(key)
							? categoricalValues[key].reduce((res, x) => {
									res[x] = x;
									return res;
							  }, {})
							: undefined,
						filterCellStyle: Object.keys(categoricalValues).includes(key)
							? undefined
							: { paddingTop: '32px' },
				  }
				: undefined,
		)
		.filter(val => val !== undefined);

const getDatasetDetails = id => Axios.get(`/api/datasets/${id}`);
const getDatasetSamples = id => Axios.get(`/api/datasets/${id}/retrieve_samples/`);

const Main = ({ classes, datasetId }) => {
	const [dataset, setDataset] = useState(undefined);
	const [error, setError] = useState(undefined);
	const [predicting, setPredicting] = useState(false);
	const [prediction, setPrediction] = useState(undefined);
	const [allSamples, setSamples] = useState([]);
	const [categoricalValues, setCategoricalValues] = useState({});
	const [selected, setSelected] = useState(undefined);
	const [currentValue, setCurrentValue] = useState(undefined);

	useEffect(() => {
		Axios.all([getDatasetDetails(datasetId), getDatasetSamples(datasetId)])
			.then(
				Axios.spread((details, samples) => {
					setDataset(details.data);
					setSamples(samples.data.samples);
					setCategoricalValues(samples.data.categorical_values);
				}),
			)
			.catch(_ =>
				setError(
					'Something went wrong while fetching the dataset. Please try another dataset or come back later.',
				),
			);

		return () => {
			setPrediction('');
			setSamples([]);
			setError(undefined);
			setDataset(undefined);
		};
	}, [datasetId]);

	const predictSample = () => {
		if (predicting) return;
		setPredicting(true);

		Axios.get(`/api/datasets/${datasetId}/predict_sample/`, {
			params: {
				sampleId: selected.tableData.id,
				foilClass: currentValue === selected.label ? undefined : currentValue,
			},
		})
			.then(res => {
				setPrediction(res.data);
				setPredicting(false);
			})
			.catch(err => {
				setError(
					'Something went wrong while predicting the sample. Please try another sample or come back later.',
				);
				setPredicting(false);
			});
	};

	return (
		<div style={styles.container}>
			{allSamples.length > 0 ? (
				<React.Fragment>
					<MaterialTable
						title={dataset.title}
						data={allSamples}
						options={{
							pageSize: 10,
							pageSizeOptions: [5, 10, 20, 100],
							maxBodyHeight: '55vh',
							filtering: true,
							rowStyle: rowData => ({
								backgroundColor:
									selected != null && selected.tableData.id === rowData.tableData.id && '#3f51b5',
								color:
									selected != null && selected.tableData.id === rowData.tableData.id && 'white',
							}),
						}}
						style={{ width: '95vw' }}
						columns={getColumns(Object.keys(allSamples[0]), categoricalValues)}
						onRowClick={(_, selectedRow) => setSelected(selectedRow)}
					/>
					<div style={styles.controls}>
						<Button
							variant="contained"
							color="primary"
							onClick={() => predictSample()}
							className={classes.predictButton}
							disabled={!selected}
						>
							{predicting ? (
								<CircularProgress size={26} color="white" />
							) : (
								<Typography>Predict sample!</Typography>
							)}
						</Button>
						<FormControl style={{ width: '15rem' }} disabled={!selected}>
							<InputLabel>Foil class</InputLabel>
							<Select value={currentValue} onChange={event => setCurrentValue(event.target.value)}>
								{categoricalValues.label &&
									categoricalValues.label
										.filter(label => selected && label !== selected.label)
										.map(label => (
											<MenuItem key={label} value={label}>
												{label}
											</MenuItem>
										))}
							</Select>
						</FormControl>
					</div>
					<Typography align="center" variant="h4">
						{predicting
							? 'Predicting...'
							: prediction ||
							  'Select a sample to predict and (optionally) the class with which it should be compared.'}
					</Typography>
				</React.Fragment>
			) : (
				<CircularProgress />
			)}
			<Snackbar open={error != null} autoHideDuration={6000} onClose={() => setError(undefined)}>
				<Alert onClose={() => setError(undefined)} severity="error">
					{error}
				</Alert>
			</Snackbar>
		</div>
	);
};
export default withStyles(styles)(Main);
