from autotest import autotest

at = autotest(model, device)
at.run_test(test_loader, 'natural', epsilon, batch_size)
at.run_test(test_loader, 'adv', epsilon, batch_size)