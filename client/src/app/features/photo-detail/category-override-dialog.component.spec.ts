import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { CategoryOverrideDialogComponent } from './category-override-dialog.component';

describe('CategoryOverrideDialogComponent', () => {

  let component: any;
  let get: ReturnType<typeof vi.fn>;
  let post: ReturnType<typeof vi.fn>;
  let dialogClose: ReturnType<typeof vi.fn>;
  let snackOpen: ReturnType<typeof vi.fn>;

  function build(currentCategory: string | null = 'silhouette', paths?: string[]) {
    get = vi.fn(() => of({ categories: [{ name: 'silhouette' }, { name: 'sports' }, { name: 'fashion' }] }));
    post = vi.fn(() => of({ success: true, path: '/a.jpg', old_category: 'silhouette', new_category: 'sports', aggregate: 7.42 }));
    dialogClose = vi.fn();
    snackOpen = vi.fn();
    TestBed.configureTestingModule({
      providers: [
        { provide: ApiService, useValue: { get, post } },
        { provide: MatSnackBar, useValue: { open: snackOpen } },
        { provide: I18nService, useValue: { t: (k: string) => k } },
        { provide: MatDialogRef, useValue: { close: dialogClose } },
        { provide: MAT_DIALOG_DATA, useValue: paths ? { paths } : { path: '/a.jpg', currentCategory } },
      ],
    });
    component = TestBed.runInInjectionContext(() => new CategoryOverrideDialogComponent());
  }

  it('loads the configured categories and preselects the current one', async () => {
    build('sports');
    await component.ngOnInit();

    expect(get).toHaveBeenCalledWith('/config/category_priorities');
    expect(component.categories()).toEqual(['silhouette', 'sports', 'fashion']);
    expect(component.selectedCategory()).toBe('sports');
    expect(component.loading()).toBe(false);
  });

  it('falls back to the first category when no current category is set', async () => {
    build(null);
    await component.ngOnInit();

    expect(component.selectedCategory()).toBe('silhouette');
  });

  it('saves the override and closes with the new category and recomputed aggregate', async () => {
    build('silhouette');
    await component.ngOnInit();
    component.selectedCategory.set('sports');

    await component.save();

    expect(post).toHaveBeenCalledWith('/comparison/override_category', { path: '/a.jpg', category: 'sports' });
    expect(dialogClose).toHaveBeenCalledWith({ category: 'sports', aggregate: 7.42 });
  });

  it('does not close on save error', async () => {
    build('silhouette');
    await component.ngOnInit();
    post.mockReturnValueOnce(throwError(() => new Error('boom')));

    await component.save();

    expect(dialogClose).not.toHaveBeenCalled();
    expect(component.saving()).toBe(false);
  });

  it('requires an explicit category choice for multiple photos', async () => {
    build(null, ['/a.jpg', '/b.jpg']);
    await component.ngOnInit();
    expect(component.selectedCategory()).toBe('');
    await component.save();
    expect(post).not.toHaveBeenCalled();
  });

  it('saves the whole selection in one request and reports the actual updated count', async () => {
    build(null, ['/a.jpg', '/b.jpg']);
    await component.ngOnInit();
    component.selectedCategory.set('sports');
    post.mockReturnValueOnce(of({ new_category: 'sports', updated: 2 }));

    await component.save();

    expect(post).toHaveBeenCalledExactlyOnceWith('/comparison/override_categories', {
      paths: ['/a.jpg', '/b.jpg'], category: 'sports',
    });
    expect(dialogClose).toHaveBeenCalledWith({ category: 'sports', updated: 2 });
  });

  it('prevents closing or duplicate submissions while a batch is saving, then permits retry', async () => {
    build(null, ['/a.jpg', '/b.jpg']);
    await component.ngOnInit();
    component.selectedCategory.set('sports');
    const response = new Subject();
    post.mockReturnValueOnce(response);
    const pending = component.save();
    const ref = TestBed.inject(MatDialogRef);
    expect(ref.disableClose).toBe(true);
    await component.save();
    expect(post).toHaveBeenCalledTimes(1);
    response.error(new Error('failed'));
    await pending;
    expect(ref.disableClose).toBe(false);
    expect(component.saving()).toBe(false);
    expect(dialogClose).not.toHaveBeenCalled();

    post.mockReturnValueOnce(of({ new_category: 'sports', updated: 2 }));
    await component.save();
    expect(dialogClose).toHaveBeenCalledWith({ category: 'sports', updated: 2 });
  });

  it('does not save if the configured categories could not be loaded', async () => {
    build('sports');
    get.mockReturnValueOnce(throwError(() => new Error('failed')));
    await component.ngOnInit();
    await component.save();
    expect(post).not.toHaveBeenCalled();
  });
});
